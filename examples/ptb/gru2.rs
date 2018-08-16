struct GatedRecurrentUnit {
    hidden0: Idx,
    hidden0: Idx,
    forget_inp: Idx,
    forget_hid: Idx,
    update_inp: Idx,
    update_hid: Idx,
    rememb_inp: Idx,
    rememb_hid: Idx,
}
impl GatedRecurrentUnit {
    /// Register the params for one gated recurrent unit
    fn new(g: &mut Graph, batch_size: usize, seq_in_dim: usize, hidden_dim: usize) -> Self {
        GatedRecurrentUnit {
            // TODO hidden0 should be Ix2 but we add batch_size dim because im lazy
            // ideally there should be an op that stacks hidden0 batch_size times
            hidden0: g.param(&[batch_size, hidden_dim]),
            forget_inp: g.param(&[input_dim, hidden_dim]),
            forget_hid: g.param(&[hidden_dim, hidden_dim]),
            update_inp: g.param(&[input_dim, hidden_dim]),
            update_hid: g.param(&[hidden_dim, hidden_dim]),
            rememb_inp: g.param(&[input_dim, hidden_dim]),
            rememb_hid: g.param(&[hidden_dim, hidden_dim]),
        }
    }
    /// Add an instance of the gated recurrent unit
    fn add_cell(&self, g: &mut Graph, hidden_in: Idx, seq_in: Idx) -> Idx {
        // TODO I think typically these are weighted and added not appended then weighted

        // Forget gate
        let fi = g.matmul(self.forget_inp, seq_in);
        let fh = g.matmul(self.forget_hid, hidden_in);
        let fa = g.add(fi, fh);
        let f = g.sigmoid(fa);

        // Remember gate
        let ri = g.matmul(self.remem_inp, seq_in);
        let rh = g.matmul(self.remem_hid, hidden_in);
        let ra = g.add(ri, rh);
        let r = g.sigmoid(ra);

        // Update gate
        let mh = g.mult(r, hidden_in)
        let ui = g.matmul(self.update_inp, seq_in);
        let uh = g.matmul(self.update_hid, mh);
        let ua = g.add(ui, uh);
        let u = g.tanh(ua);

        // New hidden
        g.op(ConvexCombine(), & [u, hidden_in, f]);
    }
}
