use std::collections::HashMap;

use tract_nnef::tract_core::ops::element_wise::ElementWiseOp;
use tract_nnef::tract_core::ops::identity::Identity;
use tract_nnef::tract_core::ops::konst::Const;
use tract_nnef::tract_core::ops::math::Add;
use tract_nnef::tract_core::ops::matmul::mir_quant::QParamKind;
use tract_nnef::tract_core::ops::nn::Sigmoid;
use tract_onnx::prelude::translator::Translate;
use tract_onnx::prelude::*;
use tract_onnx::tract_core::ops::binary::UnaryOp;
use tract_onnx::tract_core::ops::matmul::mir_quant_unary::QMatMulUnary;
use tract_onnx::tract_core::ops::matmul::{MatMulQParams, MatMulUnary};
use tract_onnx::tract_core::ops::source::TypedSource;
use tract_onnx::tract_hir::ops::cnn::ConvUnary;

#[derive(Debug)]
pub struct QuantTranslator;

impl Translate<TypedFact, Box<dyn TypedOp>, TypedFact, Box<dyn TypedOp>> for QuantTranslator {
    fn translate_node(
        &self,
        _source: &Graph<TypedFact, Box<dyn TypedOp>>,
        node: &Node<TypedFact, Box<dyn TypedOp>>,
        target: &mut Graph<TypedFact, Box<dyn TypedOp>>,
        mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        let new_op: Box<dyn TypedOp> = if let Some(source) = node.op_as::<TypedSource>() {
            Box::new(TypedSource::new(fact_f32_to_q(&source.fact)))
        } else if let Some(op) = node.op_as::<ConvUnary>() {
            Box::new(ConvUnary {
                kernel: tensor_f32_to_q(&op.kernel),
                bias: op.bias.as_ref().map(tensor_f32_to_q),
                q_params: Some((qdt(), MatMulQParams::noop_static(u8::datum_type()))),
                ..op.clone()
            })
        } else if let Some(op) = node.op_as::<MatMulUnary>() {
            let mut qp = MatMulQParams::all_from_qtype();
            qp.c0 = QParamKind::Attr(rctensor0(128i32));
            qp.c_scale = QParamKind::Attr(rctensor0(0.01));
            Box::new(QMatMulUnary {
                a: tensor_f32_to_q(&op.a),
                a_trans: op.a_trans,
                b_trans: op.b_trans,
                c_trans: op.c_trans,
                output_type: qdt(),
                params: qp,
                bias: None,
            })
        } else if let Some(op) = node.op_as::<Const>() {
            let a = tensor_f32_to_q(&op.0);
            Box::new(Const::new(a))
        } else if let Some(op) = node.op_as::<ElementWiseOp>() {
            if op.0.is::<Sigmoid>() {
                let fact = target.outlet_fact(mapping[&node.inputs[0]])?.datum_type;
                let rank = target.outlet_fact(mapping[&node.inputs[0]])?.rank();
                let t = rctensor0(0.0f32)
                    .cast_to_dt(fact)?
                    .into_owned()
                    .broadcast_into_rank(rank)?
                    .into_arc_tensor();
                Box::new(tract_onnx::tract_core::ops::math::max::unary(t))
            } else {
                node.op.clone() as Box<dyn TypedOp>
            }
        } else if let Some(op) = node.op_as::<UnaryOp>() {
            if op.mini_op.is::<Add>() {
                let prec = target.node(mapping[&node.inputs[0]].node);
                // ignore bias
                if prec.op_is::<QMatMulUnary>() {
                    /*
                    let mut mm = prec.op_as::<QMatMulUnary>().unwrap().clone();
                    let mut a = op.a.clone().into_tensor();
                    a.remove_axis(0)?;
                    a.remove_axis(1)?;
                    a.as_slice_mut::<f32>()?
                        .iter_mut()
                        .for_each(|x| *x *= qdt().zp_scale().1.powi(2));
                    mm.bias = Some(a.cast_to::<i32>()?.clone().into_owned().into_arc_tensor());
                    input_override = Some(tvec!(mapping[&prec.inputs[0]]));
                    Box::new(mm) as Box<dyn TypedOp>;
                    */
                    Box::new(Identity)
                } else if prec.op_is::<ConvUnary>() {
                    /*
                    let mut mm = prec.op_as::<ConvUnary>().unwrap().clone();
                    let mut a = op.a.clone().into_tensor();
                    a.as_slice_mut::<f32>()?
                        .iter_mut()
                        .for_each(|x| *x *= qdt().zp_scale().1.powi(2));
                    a.remove_axis(0)?;
                    a.remove_axis(1)?;
                    mm.bias = Some(a.cast_to::<i32>()?.clone().into_owned().into_arc_tensor());
                    input_override = Some(tvec!(mapping[&prec.inputs[0]]));
                    Box::new(mm) as Box<dyn TypedOp>
                    */
                    Box::new(Identity)
                } else {
                    let mut new = op.clone();
                    new.a = tensor_f32_to_q(&op.a);
                    Box::new(new)
                }
            } else {
                let mut new = op.clone();
                new.a = tensor_f32_to_q(&op.a);
                Box::new(new)
            }
        } else {
            node.op.clone()
        };
        target.wire_node(
            &node.name,
            new_op,
            &node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>(),
        )
    }
}

fn qdt() -> DatumType {
    DatumType::QU8(QParams::ZpScale { zero_point: 128, scale: 0.01 })
}

fn fact_f32_to_q(t: &TypedFact) -> TypedFact {
    if t.datum_type == f32::datum_type() {
        let mut t = t.clone();
        t.datum_type = qdt();
        t
    } else {
        t.clone()
    }
}

fn tensor_f32_to_q(t: &Arc<Tensor>) -> Arc<Tensor> {
    use std::cmp::Ordering::*;
    if t.datum_type() == f32::datum_type() {
        let min = t
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .min_by(|a, b| if a < b { Less } else { Greater })
            .unwrap()
            .max(0.);
        let max = t
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .max_by(|a, b| if a < b { Less } else { Greater })
            .unwrap()
            .min(0.);
        let scale = (max - min) / 255.;
        let zero_point = ((-min) / scale) as i32;
        let qp = DatumType::QU8(QParams::ZpScale { zero_point, scale });
        let new = t.cast_to_dt(qp).unwrap().into_owned().into_arc_tensor();
        new
    } else {
        Arc::clone(t)
    }
}
