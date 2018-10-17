extern crate ocl;
use std::collections::BTreeMap;

struct Idx(usize);


enum OperationHandle {
    Matmul(matrix: Idx, bias:Idx, arg: Idx, value: Idx)
}

// every operation needs to have associated
// * forward kernels
// * backward kernels
// * parameters
// * OperationHandle (to allow for deletion or smth)
// * Ideally these operations will check their input shape / types and fail appropriately
enum Operation {
    Matmul, Conv, GlobalPool, Add, Embedding,
}

struct OperationBuilder {
    initializer: ???,
    type: Operation,
    // other params to be set, inputs, etc
    // maybe different builders for each kind of operation?
    // should take Idx to reuse weights
}


// ocl::Buffer<T> and ndarray::Array<Sh, T> are heterogenous types, we use enums so we can keep
// them in the same container. Maybe we can ge intelligent and do type and shape checking while
// building the graph. Honestly, this abstraction probably deserves to be its own library linking
// ocl to ndarray. Note that ocl buffers go up to 3 dimensions so some care has to be taken when
// translating between ndarray and the ocl buffer / image / etc.
// Consider abstract kernels that use ndarray AND ocl
struct Tensor {
    buffer: TensorOCL,
    val: TensorCPU,
    // makes sure val is up to date, set to false if kernel updating buffer is enq().
    updated: bool,
}
enum TensorCPU {
    Dynamicf32<ArrayD<f32>>, //etc
}
enum TensorOCL {
    B4f32<ocl::Buffer<f32>, [usize; 4]>, // etc
}



enum Node {
    /// User inputed values, possibly updating every forward step because iterator
    Const(Option<Box<Iterator<Vec<f32>>>),
    Operation(Operation),
}
type Map<T> = BTreeMap<Idx, T>;

struct Graph {
    // The actual computation graph that the user cares about.
    operations: Map<Operation>,

    // OpenCL context. There should be a way of adding more control to the OCL context and perhaps
    // swapping out the backend somehow?
    // It would be nice to unify this with an ndarray implementation.
    // Further in the future, there may be a need for a distributed context as well
    proque: ocl::ProQue,

    // Name important idxs so you can retrieve them after saving... maybe allow naming of handles?
    // really every Idx should be somehow retrievable after saving
    named_idxs: BTreeMap<String, Idx>

    values: Map<Tensor>,
    gradients: Map<Tensor>,
    optimizer_instances: Map<Tensor>,

    // Computation kernels that execute on the OCL device.
    forward_kernels: Map<ocl::Kernel>,
    backward_kernels: Map<ocl::Kernel>,
    optimizer_kernels: Map<ocl::Kernel>,
}

// TODO consider using ndarray and results
impl Graph {
    fn new() -> Self;

    // Instantiate the appropriate kernel and buffers
    // write the initialized parameters to their value buffers
    // maybe instantiate optimizer instances for these parameters?
    // return handles to values/gradients
    fn op<O: Into<Operation>>(&mut self, o: O) -> OperationHandle;

    fn const(&mut self, Vec<T>) -> Idx;

    fn forward1(&mut self, idx: Idx);
    fn forward(&mut self);

    fn backward1(&mut self, idx: Idx);
    fn backward(&mut self);

    fn optimize1(&mut self, idx: Idx);
    fn optimize(&mut self);

    fn get_value(&self, idx: Idx) -> Vec<T>;
    fn set_value(&mut self, idx: Idx, v: Vec<T>);
    fn get_gradient(&self, idx: Idx) -> Vec<T>;
    fn set_gradient(&mut self, idx: Idx, g: Vec<T>);
}
