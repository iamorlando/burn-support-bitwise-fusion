use super::optimization::{FusedLowerTriangularCorrelate, LowerTriangularCorrelateOptimization};
use crate::{
    engine::{
        fuser::TraceOperationFuser,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
    optim::CubeOptimization,
};
use burn_fusion::{FuserProperties, FuserStatus, OperationFuser};
use burn_ir::{BinaryOpIr, FloatOperationIr, OperationIr};
use burn_std::DType;
use cubecl::Runtime;

/// Fuses elementwise producers into a row-wise lower-triangular correlation kernel.
pub struct LowerTriangularCorrelateFuser<R: Runtime> {
    fuser: TraceOperationFuser,
    fuser_read_fallback: TraceOperationFuser,
    device: R::Device,
    correlate: Option<FusedLowerTriangularCorrelate>,
    len_stream: usize,
    fallback_pos: usize,
}

impl<R: Runtime> Clone for LowerTriangularCorrelateFuser<R> {
    fn clone(&self) -> Self {
        Self {
            fuser: self.fuser.clone(),
            fuser_read_fallback: self.fuser_read_fallback.clone(),
            device: self.device.clone(),
            correlate: self.correlate.clone(),
            len_stream: self.len_stream,
            fallback_pos: self.fallback_pos,
        }
    }
}

impl<R: Runtime> LowerTriangularCorrelateFuser<R> {
    pub fn new(device: R::Device) -> Self {
        let client = R::client(&device);
        let hardware = &client.properties().hardware;
        let max_bindings = hardware.max_bindings;
        let settings_read = FuseSettings {
            inplace: true,
            ref_layout: RefLayoutSetting::OnlyContiguous,
            broadcast: false,
            output_shape_updates: true,
            vectorization: VectorizationSetting::Deactivated,
        };
        let settings_fallback = FuseSettings::default();

        Self {
            fuser: TraceOperationFuser::new(max_bindings, settings_read),
            fuser_read_fallback: TraceOperationFuser::new(max_bindings, settings_fallback),
            device,
            correlate: None,
            len_stream: 0,
            fallback_pos: 0,
        }
    }

    fn validate(op: &BinaryOpIr) -> bool {
        let lhs = op.lhs.shape.as_slice();
        let rhs = op.rhs.shape.as_slice();
        let out = op.out.shape.as_slice();

        op.lhs.dtype == DType::F32
            && op.rhs.dtype == DType::F32
            && op.out.dtype == DType::F32
            && lhs.len() == 2
            && rhs.len() == 2
            && out == lhs
            && lhs[1] == rhs[0]
            && rhs[0] == rhs[1]
    }

    fn on_elemwise_read(&mut self, operation: &OperationIr) {
        let can_register =
            self.fuser.can_fuse(operation) && self.fuser_read_fallback.can_fuse(operation);

        if can_register {
            self.fuser.fuse(operation);
            self.fuser_read_fallback.fuse(operation);
            self.len_stream += 1;
        } else {
            self.fuser.close();
            self.fuser_read_fallback.close();
        }
    }

    fn on_correlate(&mut self, op: &BinaryOpIr) {
        if !Self::validate(op) {
            self.fuser.close();
            self.fuser_read_fallback.close();
            return;
        }

        if self.fuser.current_output_shape != op.lhs.shape {
            self.fuser.close();
            self.fuser_read_fallback.close();
            return;
        }

        if op.lhs.shape.as_slice()[1] == 0 {
            self.fuser.close();
            self.fuser_read_fallback.close();
            return;
        }

        let settings_write = FuseSettings {
            inplace: false,
            output_shape_updates: false,
            vectorization: VectorizationSetting::Deactivated,
            broadcast: false,
            ref_layout: RefLayoutSetting::OnlyContiguous,
        };

        let [independent] = self.fuser.next_block([&op.lhs], settings_write, false);
        let Some(lower) = self.fuser.input_indexed(&op.rhs) else {
            self.fuser.close();
            self.fuser_read_fallback.close();
            return;
        };
        let output = self.fuser.output_unhandled(&op.out);

        self.fallback_pos = self.len_stream;
        self.len_stream += 1;
        self.correlate = Some(FusedLowerTriangularCorrelate {
            independent,
            lower,
            output,
            op: op.clone(),
        });
        self.fuser_read_fallback.close();
        self.fuser.close();
    }
}

impl<R: Runtime> OperationFuser<CubeOptimization<R>> for LowerTriangularCorrelateFuser<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        if self.correlate.is_some() || self.fuser.status() == FuserStatus::Closed {
            return;
        }

        match operation {
            OperationIr::Float(_, FloatOperationIr::LowerTriangularCorrelate(op)) => {
                self.on_correlate(op);
            }
            OperationIr::Init(_) => {
                // Init registers its handle eagerly and does not need kernel work.
                self.len_stream += 1;
            }
            _ => self.on_elemwise_read(operation),
        }
    }

    fn finish(&mut self) -> CubeOptimization<R> {
        let client = R::client(&self.device);
        let correlate = self
            .correlate
            .as_ref()
            .expect("Lower triangular correlate fuser finished before the op was registered")
            .clone();

        let trace = self.fuser.finish();
        let trace_read_fallback = self.fuser_read_fallback.finish();

        CubeOptimization::LowerTriangularCorrelate(LowerTriangularCorrelateOptimization::new(
            trace,
            trace_read_fallback,
            client,
            self.device.clone(),
            self.len_stream,
            self.fallback_pos,
            correlate,
        ))
    }

    fn reset(&mut self) {
        self.fuser.reset();
        self.fuser_read_fallback.reset();
        self.correlate = None;
        self.len_stream = 0;
        self.fallback_pos = 0;
    }

    fn status(&self) -> FuserStatus {
        if self.correlate.is_some() {
            FuserStatus::Closed
        } else {
            self.fuser.status()
        }
    }

    fn properties(&self) -> FuserProperties {
        let mut properties = self.fuser.properties();
        properties.ready = self.correlate.is_some();

        if self.correlate.is_some() {
            // The custom kernel removes the producer output write, the correlate input read,
            // and the extra kernel launch that would exist at the boundary.
            properties.score = properties.score.saturating_add(210);
        }

        properties
    }

    fn len(&self) -> usize {
        self.len_stream
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}
