#![cfg(feature = "fusion")]

use burn_cubecl::{CubeBackend, CubeRuntime, fusion::FusionCubeRuntime};
use burn_fusion::{
    FusionRuntime, get_client,
    inspect::{BlockKind, FusionInspector},
    stream::{Operation, OperationStreams, StreamId},
};
use burn_ir::{BinaryOpIr, FloatOperationIr, HandleContainer, OperationIr, TensorIr};
use burn_tensor::{Tensor, TensorData, TensorPrimitive, backend::Backend};
use cubecl::wgpu::WgpuRuntime;
use std::sync::atomic::{AtomicU64, Ordering};

type InnerBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type TestBackend = burn_fusion::Fusion<InnerBackend>;
type TestTensor<const D: usize> = Tensor<TestBackend, D>;

#[derive(Clone, Debug)]
struct LowerTriangularCorrelateFallback;

impl<R> Operation<FusionCubeRuntime<R>> for LowerTriangularCorrelateFallback
where
    R: CubeRuntime,
{
    fn execute(
        &self,
        _handles: &mut HandleContainer<<FusionCubeRuntime<R> as FusionRuntime>::FusionHandle>,
    ) {
        panic!("lower triangular correlate should execute through the fused CubeCL kernel")
    }
}

#[test]
fn lower_triangular_correlate_fuses_producer_into_single_kernel() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();

        let base = TestTensor::<2>::from_data(
            TensorData::from([[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            &device,
        );
        let lower =
            TestTensor::<2>::from_data(TensorData::from([[2.0_f32, 0.0], [0.5, 3.0]]), &device);
        TestBackend::sync(&device).unwrap();

        let inspector = FusionInspector::install(stream);

        let independent = base.mul_scalar(2.0).add_scalar(1.0);
        let output = lower_triangular_correlate(independent, lower);

        output.into_data().assert_eq(
            &TensorData::from([[6.0_f32, 16.5], [14.0, 30.5], [22.0, 44.5], [30.0, 58.5]]),
            false,
        );
        TestBackend::sync(&device).unwrap();

        let reports = inspector.drain();
        let tables = reports
            .iter()
            .map(|report| report.format_table())
            .collect::<Vec<_>>()
            .join("\n\n");

        let block = reports
            .iter()
            .flat_map(|report| report.blocks.iter())
            .find(|block| block.fuser_name() == Some("LowerTriangularCorrelate"))
            .unwrap_or_else(|| panic!("no LowerTriangularCorrelate fused block found\n\n{tables}"));

        assert!(
            matches!(
                block.kind,
                BlockKind::Fused {
                    name: "LowerTriangularCorrelate",
                    ..
                }
            ),
            "expected LowerTriangularCorrelate fused block, got {:?}\n\n{tables}",
            block.kind,
        );
        assert!(
            block
                .operations
                .iter()
                .any(|op| format!("{op:?}").contains("LowerTriangularCorrelate")),
            "LowerTriangularCorrelate op missing from fused block\n\n{tables}",
        );
        assert!(
            !tables.contains("Custom"),
            "correlation should not appear as Custom\n\n{tables}",
        );
    });
}

#[test]
fn lower_triangular_correlate_handles_large_path_grid() {
    const PATHS: usize = 65_536;

    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let mut values = Vec::with_capacity(PATHS * 2);

        for _ in 0..PATHS {
            values.extend_from_slice(&[1.0_f32, 2.0]);
        }

        let base = TestTensor::<2>::from_data(TensorData::new(values, [PATHS, 2]), &device);
        let lower =
            TestTensor::<2>::from_data(TensorData::from([[2.0_f32, 0.0], [0.5, 3.0]]), &device);
        TestBackend::sync(&device).unwrap();

        let independent = base.mul_scalar(2.0).add_scalar(1.0);
        let output = lower_triangular_correlate(independent, lower);
        let data = output.into_data();
        let values = data.as_slice::<f32>().unwrap();
        let last = (PATHS - 1) * 2;

        assert_eq!(values.len(), PATHS * 2);
        assert_eq!(values[0], 6.0);
        assert_eq!(values[1], 16.5);
        assert_eq!(values[last], 6.0);
        assert_eq!(values[last + 1], 16.5);
        TestBackend::sync(&device).unwrap();
    });
}

fn lower_triangular_correlate(independent: TestTensor<2>, lower: TestTensor<2>) -> TestTensor<2> {
    let device = independent.device();
    let client = get_client::<InnerBackend>(&device);
    let independent = match independent.into_primitive() {
        TensorPrimitive::Float(tensor) => tensor,
        _ => unreachable!("independent tensor should be float"),
    };
    let lower = match lower.into_primitive() {
        TensorPrimitive::Float(tensor) => tensor,
        _ => unreachable!("lower tensor should be float"),
    };
    let streams = OperationStreams::with_inputs([&independent, &lower]);
    let lhs = independent.into_ir();
    let rhs = lower.into_ir();
    let out = TensorIr::uninit(client.create_empty_handle(), lhs.shape.clone(), lhs.dtype);
    let desc = BinaryOpIr { lhs, rhs, out };

    let mut outputs = client.register(
        streams,
        OperationIr::Float(
            desc.out.dtype,
            FloatOperationIr::LowerTriangularCorrelate(desc),
        ),
        LowerTriangularCorrelateFallback,
    );
    let output = outputs
        .pop()
        .expect("lower triangular correlate should register one output");

    Tensor::from_primitive(TensorPrimitive::Float(output))
}

fn test_stream() -> StreamId {
    static COUNTER: AtomicU64 = AtomicU64::new(10_000);
    StreamId {
        value: COUNTER.fetch_add(1, Ordering::Relaxed),
    }
}
