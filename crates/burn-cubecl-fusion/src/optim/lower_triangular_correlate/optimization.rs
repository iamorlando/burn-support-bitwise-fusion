use crate::{
    CubeFusionHandle, FallbackOperation,
    engine::{
        codegen::{
            io::{input_as_slice, ref_shape},
            ir::{
                FuseArg, FuseBlockConfig, GlobalArgs, GlobalArgsLaunch, multi_block_variables_init,
            },
            kernel::{fuse_on_read, fuse_on_write, init_locals},
        },
        launch::{
            FuseTraceLauncher,
            runner::{TraceRunner, Vectorization},
        },
        trace::{FuseTrace, TraceError, TuneOutput},
    },
    optim::elemwise::ElemwiseRunner,
};
use burn_fusion::stream::Context;
use burn_ir::BinaryOpIr;
use cubecl::{CubeDim, Runtime, calculate_cube_count_elemwise, client::ComputeClient, prelude::*};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Fused lower-triangular correlation operation.
pub struct LowerTriangularCorrelateOptimization<R: Runtime> {
    info: Arc<LowerTriangularCorrelateOptimizationInfo<R>>,
}

struct LowerTriangularCorrelateOptimizationInfo<R: Runtime> {
    trace: FuseTrace,
    trace_read_fallback: FuseTrace,
    client: ComputeClient<R>,
    device: R::Device,
    len: usize,
    fallback_pos: usize,
    correlate: FusedLowerTriangularCorrelate,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LowerTriangularCorrelateOptimizationState {
    trace: FuseTrace,
    trace_read_fallback: FuseTrace,
    correlate: FusedLowerTriangularCorrelate,
    len: usize,
    fallback_pos: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FusedLowerTriangularCorrelate {
    pub(crate) independent: FuseArg,
    pub(crate) lower: FuseArg,
    pub(crate) output: FuseArg,
    pub(crate) factors: usize,
    pub(crate) op: BinaryOpIr,
}

pub struct LowerTriangularCorrelateTuneArg<R: Runtime> {
    info: Arc<LowerTriangularCorrelateOptimizationInfo<R>>,
    fallback: Box<dyn FallbackOperation<R>>,
}

#[derive(new)]
pub struct FusedLowerTriangularCorrelateLaunch<'a> {
    correlate: &'a FusedLowerTriangularCorrelate,
}

impl<R: Runtime> LowerTriangularCorrelateTuneArg<R> {
    fn execute_fused(
        &self,
        context: &mut Context<CubeFusionHandle<R>>,
    ) -> Result<TuneOutput<R>, TraceError<LaunchError>> {
        let launch = FusedLowerTriangularCorrelateLaunch::new(&self.info.correlate);
        let launcher = FuseTraceLauncher::new(&self.info.trace, &launch);

        launcher.launch(&self.info.client, &self.info.device, context)
    }

    fn execute_fallback(&self, context: &mut Context<CubeFusionHandle<R>>) -> TuneOutput<R> {
        let launcher = FuseTraceLauncher::new(&self.info.trace_read_fallback, &ElemwiseRunner);
        let output_read = launcher
            .launch(&self.info.client, &self.info.device, context)
            .unwrap();

        self.fallback.run(context);
        output_read
    }
}

impl<R: Runtime> LowerTriangularCorrelateOptimization<R> {
    pub fn new(
        trace: FuseTrace,
        trace_read_fallback: FuseTrace,
        client: ComputeClient<R>,
        device: R::Device,
        len: usize,
        fallback_pos: usize,
        correlate: FusedLowerTriangularCorrelate,
    ) -> Self {
        Self {
            info: Arc::new(LowerTriangularCorrelateOptimizationInfo {
                trace,
                trace_read_fallback,
                client,
                device,
                len,
                fallback_pos,
                correlate,
            }),
        }
    }

    pub fn execute(
        &mut self,
        context: &mut Context<CubeFusionHandle<R>>,
        fallback: impl FnOnce(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        let arg = LowerTriangularCorrelateTuneArg {
            info: self.info.clone(),
            fallback: fallback(self.info.fallback_pos),
        };

        if arg.execute_fused(context).is_err() {
            arg.execute_fallback(context);
        }
    }

    pub fn num_ops_fused(&self) -> usize {
        self.info.len
    }

    pub fn to_state(&self) -> LowerTriangularCorrelateOptimizationState {
        LowerTriangularCorrelateOptimizationState {
            trace: self.info.trace.clone(),
            trace_read_fallback: self.info.trace_read_fallback.clone(),
            correlate: self.info.correlate.clone(),
            len: self.info.len,
            fallback_pos: self.info.fallback_pos,
        }
    }

    pub fn from_state(
        device: &R::Device,
        state: LowerTriangularCorrelateOptimizationState,
    ) -> Self {
        Self {
            info: Arc::new(LowerTriangularCorrelateOptimizationInfo {
                trace: state.trace,
                trace_read_fallback: state.trace_read_fallback,
                client: R::client(device),
                device: device.clone(),
                len: state.len,
                fallback_pos: state.fallback_pos,
                correlate: state.correlate,
            }),
        }
    }
}

impl<R: Runtime> Vectorization<R> for FusedLowerTriangularCorrelateLaunch<'_> {}

impl<R: Runtime> TraceRunner<R> for FusedLowerTriangularCorrelateLaunch<'_> {
    type Error = LaunchError;

    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R>,
        inputs: GlobalArgsLaunch<R>,
        outputs: GlobalArgsLaunch<R>,
        configs: &'a [FuseBlockConfig],
    ) -> Result<(), Self::Error> {
        let [config_read, config_write] = [&configs[0], &configs[1]];
        let shape = outputs.shape_ref(&config_write.ref_layout, config_write.rank);
        let working_units = shape[0];
        let cube_dim = CubeDim::new(client, working_units);
        let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
        let address_type = inputs
            .required_address_type()
            .max(outputs.required_address_type());

        unsafe {
            lower_triangular_correlate_fused::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                address_type,
                inputs,
                outputs,
                config_read.clone(),
                config_write.clone(),
                self.correlate.independent.clone(),
                self.correlate.lower.clone(),
                self.correlate.output.clone(),
                self.correlate.factors,
            );
        }

        Ok(())
    }
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn lower_triangular_correlate_fused(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    #[comptime] config_read: &FuseBlockConfig,
    #[comptime] config_write: &FuseBlockConfig,
    #[comptime] independent: FuseArg,
    #[comptime] lower: FuseArg,
    #[comptime] output: FuseArg,
    #[comptime] factors: usize,
) {
    multi_block_variables_init(config_read, &mut outputs.variables);
    multi_block_variables_init(config_write, &mut outputs.variables);

    let mut locals_read = init_locals(inputs, outputs, config_read);
    let mut locals_write = init_locals(inputs, outputs, config_write);
    let paths = ref_shape(&locals_write, 0);
    let pos = ABSOLUTE_POS;

    if pos < paths {
        let p = pos;
        let mut acc = Array::new(factors);
        let lower_values = lower_as_slice(inputs, lower);

        let mut j = 0;
        while j < factors {
            acc[j] = 0.0f32;
            j += 1;
        }

        let mut k = 0;
        while k < factors {
            let read_pos = p * factors + k;
            let z = fuse_on_read::<f32, Const<1>>(
                inputs,
                outputs,
                &mut locals_read,
                read_pos,
                comptime! {
                    let mut sequence = Sequence::new();
                    sequence.push(independent.clone());
                    sequence
                },
                config_read,
            )[0][0];

            j = k;
            while j < factors {
                let l = lower_values[j * factors + k];
                acc[j] += z * l;
                j += 1;
            }

            k += 1;
        }

        j = 0;
        while j < factors {
            let write_pos = p * factors + j;
            let mut values = Registry::<FuseArg, Vector<f32, Const<1>>>::new();
            let mut args = comptime![Vec::<FuseArg>::new()];
            values.insert(comptime![output.clone()], Vector::new(acc[j]));
            comptime![args.push(output.clone())];

            fuse_on_write::<f32, Const<1>>(
                inputs,
                outputs,
                &mut locals_write,
                write_pos,
                values,
                args,
                config_write,
            );

            j += 1;
        }
    }
}

#[cube]
fn lower_as_slice(inputs: &GlobalArgs, #[comptime] lower: FuseArg) -> Slice<f32> {
    match lower {
        FuseArg::Input(pos, ..) => input_as_slice::<f32>(inputs, pos),
        _ => comptime![panic!(
            "Lower triangular correlation expects lower to be an input"
        )],
    }
}
