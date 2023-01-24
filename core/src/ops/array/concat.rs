use tract_data::itertools::Itertools;

use crate::internal::*;

use super::Slice;

#[derive(new, Debug, Clone, Hash)]
pub struct TypedConcat {
    pub axis: usize,
}
impl_dyn_hash!(TypedConcat);

impl TypedConcat {
    pub fn offsets(&self, inputs: &[&TypedFact]) -> TractResult<Vec<TDim>> {
        let mut offsets = vec![0.to_dim()];
        for slice in inputs {
            let len = slice.shape[self.axis].clone();
            let offset = len + offsets.last().unwrap();
            offsets.push(offset)
        }
        Ok(offsets)
    }
}

impl Op for TypedConcat {
    fn name(&self) -> Cow<str> {
        "Concat".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}", self.axis)])
    }

    op_as_typed_op!();
}

impl TypedOp for TypedConcat {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].without_value();
        for input in inputs {
            if input.rank() != fact.rank()
                || input
                    .shape
                    .iter()
                    .zip(fact.shape.iter())
                    .enumerate()
                    .filter(|(ax, _)| *ax != self.axis)
                    .any(|(_, (i, f))| i != f)
            {
                bail!("Inconsistent concat {:?} inputs: {:?}", self, inputs);
            }
        }
        fact.shape.set(self.axis, self.offsets(inputs)?.pop().unwrap());
        Ok(tvec!(fact))
    }

    fn invariants(&self, inputs: &[&TypedFact], outputs: &[&TypedFact]) -> TractResult<Invariants> {
        let rank = inputs[0].rank();
        (0..rank)
            .filter(|&ax| ax != self.axis)
            .map(|axis| AxisInfo::for_facts(inputs, outputs, axis))
            .collect()
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let axis =
            if let Some(axis) = change.transform_axis(self.axis) { axis } else { return Ok(None) };
        let op = TypedConcat { axis };
        Ok(Some(AxisChangeConsequence::new(model, node, Some(Box::new(op)), change)))
    }

    fn slice(
        &self,
        patch: &mut TypedModelPatch,
        prefix: &str,
        inputs: &[OutletId],
        output_axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<TVec<OutletId>>> {
        let facts =
            inputs.iter().map(|o| patch.outlet_fact(*o)).collect::<TractResult<TVec<_>>>()?;
        let offsets = self.offsets(&facts)?;
        std::mem::forget(facts);
        for (slice_start, slice_end) in offsets.iter().tuple_windows() {
            if let (Ok(slice_start), Ok(slice_end)) = (slice_start.to_usize(), slice_end.to_usize())
            {
                if slice_start <= start && end <= slice_end {
                    return patch
                        .wire_node(
                            format!("{prefix}.slice-{output_axis}.{start}..{end}"),
                            Slice {
                                axis: output_axis,
                                start: (start - slice_start).to_dim(),
                                end: (end - slice_start).to_dim(),
                            },
                            &inputs,
                        )
                        .map(Some);
                }
            }
        }
        Ok(None)
    }
}

impl EvalOp for TypedConcat {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let result = Tensor::stack_tensors(self.axis, &inputs)?;
        Ok(tvec![result.into_tvalue()])
    }
}
