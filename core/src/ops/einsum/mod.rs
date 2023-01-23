use std::fmt::Debug;

use crate::internal::*;

mod eval;
mod expr;
pub use expr::Axis;
pub use expr::Expr;
mod to_matmul;

#[derive(Clone, Hash, new)]
pub struct EinSum {
    pub expr: Expr,
    pub operating_dt: DatumType,
}

impl_dyn_hash!(EinSum);

impl Debug for EinSum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EinSum {} ({:?})", self.expr, self.operating_dt)
    }
}

impl Op for EinSum {
    fn name(&self) -> Cow<str> {
        "EinSum".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{}", self.expr)])
    }

    op_as_typed_op!();
}

impl EvalOp for EinSum {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        dispatch_numbers!(eval::eval_t(self.operating_dt)(&self.expr, inputs))
    }
}

impl TypedOp for EinSum {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs
            .iter()
            .enumerate()
            .all(|(ix, fact)| fact.rank() == self.expr.input_rank(ix)));
        let shapes: TVec<&[TDim]> = inputs.iter().map(|t| &*t.shape).collect();
        Ok(tvec!(TypedFact::dt_shape(self.operating_dt, eval::output_shape(&self.expr, &shapes))))
    }

    fn invariants(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<Invariants> {
        let inv = self
            .expr
            .iter_all_axes()
            .filter_map(|axis| {
                // if axis is used twice, don't even dare do anything
                if axis.inputs.iter().any(|input| input.len() > 1) {
                    None
                } else {
                    let i = (0..inputs.len()).map(|i| axis.inputs[i].get(0).cloned()).collect();
                    let o = axis.result;
                    Some(AxisInfo { inputs: i, outputs: tvec!(o), period: 1, disposable: true })
                }
            })
            .collect();
        Ok(inv)
    }

    fn cost(&self, inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let shapes: TVec<&[TDim]> = inputs.iter().map(|t| &*t.shape).collect();
        let oshape = eval::output_shape(&self.expr, &shapes);
        let ks = self
            .expr
            .sum
            .iter()
            .map(|axis| {
                axis.inputs
                    .iter()
                    .enumerate()
                    .flat_map(|(ix, axes)| {
                        axes.iter()
                            .map(|axis| shapes[ix][*axis].clone())
                            .collect::<TVec<_>>()
                            .into_iter()
                    })
                    .max()
                    .unwrap()
            })
            .product::<TDim>();
        Ok(tvec!((Cost::FMA(self.operating_dt), oshape.iter().product::<TDim>() * ks)))
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        prefix: &str,
        inputs: &[OutletId],
        _output_axis: usize,
        _start: usize,
        _end: usize,
    ) -> TractResult<Option<TVec<OutletId>>> {
        patch.wire_node(prefix, self.clone(), inputs).map(Some)
    }

    #[allow(unused_variables)]
    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let (mut inputs, mut result) = self.expr.to_strs();
        let interface: &mut String = match io {
            InOut::In(i) => &mut inputs[i],
            InOut::Out(o) => &mut result,
        };
        let mut axes: Vec<char> = interface.chars().collect();
        match change {
            AxisOp::Rm(rm) => {
                axes.remove(*rm);
            }
            AxisOp::Add(add) => axes.insert(*add, self.expr.available_label()),
            AxisOp::Move(from, to) => {
                let c = axes.remove(*from);
                axes.insert(*to, c);
            }
            _ => return Ok(None),
        };
        *interface = axes.into_iter().collect();
        let expr = Expr::from_strs(&inputs, Some(&result));
        return Ok(Some(AxisChangeConsequence {
            substitute_op: Some(Box::new(EinSum::new(expr, self.operating_dt))),
            wire_changes: tvec!((io, change.clone())),
        }));
    }

    /*
    fn declutter(
    &self,
    model: &TypedModel,
    node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
    to_matmul::declutter(self, model, node)
    }
    */

    as_op!();
}
