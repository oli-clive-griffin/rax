use rusty_grad::backward::{accum_grads, GradMap};
use rusty_grad::node::Node;
use rusty_grad::ops::{add, mul, mmul, relu, sqr, sub};
use rusty_grad::optimizer::{Optimizer, ParamsMap, SGD};
use rusty_grad::tensor::Tensor;
use std::collections::HashMap;
use std::rc::Rc;
use std::thread::sleep;
use std::time::Duration;

#[test]
fn test_train() {
    // a dead simple MLP. the model is a pure forward pass function,
    // without having to worry about stateful parameter handling.
    fn model(params: &ParamsMap, x: Tensor) -> Rc<Node> {
        let w1 = Rc::new(Node::TensorParam(params.0.get("w1").unwrap().clone(), "w1"));
        let b1 = Rc::new(Node::TensorParam(params.0.get("b1").unwrap().clone(), "b1"));
        let w2 = Rc::new(Node::TensorParam(params.0.get("w2").unwrap().clone(), "w2"));
        let b2 = Rc::new(Node::TensorParam(params.0.get("b2").unwrap().clone(), "b2"));
        let w3 = Rc::new(Node::TensorParam(params.0.get("w3").unwrap().clone(), "w3"));
        let b3 = Rc::new(Node::TensorParam(params.0.get("b3").unwrap().clone(), "b3"));

        let input = Rc::new(Node::TensorParam(x, "input"));

        let x1 = relu(add(mmul(input, w1), b1));
        // println!("x1: {:?}", x1.val()); // print the tensor, not the node + subtree

        let x2 = relu(add(mmul(x1, w2), b2));
        // println!("x2: {:?}", x2.val()); // print the tensor, not the node + subtree

        let x3 = relu(add(mmul(x2, w3), b3));
        // println!("x3: {:?}", x3.val()); // print the tensor, not the node + subtree

        if x3.val().item().unwrap() - 0.01 < 1e-3 {
            println!("x3: {:.4}", x3.val().item().unwrap());
        }

        return x3;
    }

    fn forward(params: &ParamsMap, input: Tensor, label: Tensor) -> (Tensor, GradMap) {
        let label = Rc::new(Node::TensorParam(label, "label"));
        let out = model(params, input);

        // let loss = mean(sq(sub(act3, label)));
        let loss = sqr(sub(out.clone(), label.clone()));
        // println!("out: {:?}, label: {:?}", out.val().item().unwrap(), label.val().item().unwrap());
        // println!("loss: {:?}", loss.val().item().unwrap()); // print the tensor, not the node + subtree

        let graph = loss.backwards();
        let grads_map = accum_grads(graph);

        (loss.val(), grads_map)
    }

    fn train_model() -> ParamsMap {
        let mut params = ParamsMap(HashMap::from([
            ("w1".to_string(), Tensor::rand(&vec![3, 4])),
            ("b1".to_string(), Tensor::rand(&vec![4])),
            ("w2".to_string(), Tensor::rand(&vec![4, 3])),
            ("b2".to_string(), Tensor::rand(&vec![3])),
            ("w3".to_string(), Tensor::rand(&vec![3, 1])),
            ("b3".to_string(), Tensor::rand(&vec![1])),
        ]));

        let optim = SGD::default();

        let x = Tensor::rand(&vec![1, 3]);
        let y = Tensor::rand(&vec![1, 1]);

        loop {
            let (loss, grads_map) = forward(&params, x.clone(), y.clone());
            params = optim.update(params, grads_map);
            sleep(Duration::from_millis(50));
            if loss.item().unwrap() < 1e-6 {
                break;
            }
        }

        return params;
    }

    println!("Training model...");
    let params = train_model();
    println!("{:?}", params);
}



#[test]
fn test__train_simple() {
    // a dead simple MLP. the model is a pure forward pass function,
    // without having to worry about stateful parameter handling.
    fn model(params: &ParamsMap) -> Rc<Node> {
        let w1 = Rc::new(Node::TensorParam(params.0.get("w1").unwrap().clone(), "w1"));
        let w2 = Rc::new(Node::TensorParam(params.0.get("w2").unwrap().clone(), "w2"));

        add(w1, w2)
    }

    fn forward(params: &ParamsMap) -> (Tensor, GradMap) {
        let out = model(params);
        let graph = out.backwards();
        let grads_map = accum_grads(graph);
        (out.val(), grads_map)
    }

    fn train_model() -> ParamsMap {
        let mut params = ParamsMap(HashMap::from([
            ("w1".to_string(), Tensor::rand(&vec![1])),
            ("w2".to_string(), Tensor::rand(&vec![1])),
        ]));

        let optim = SGD::default();

        loop {
            let (loss, grads_map) = forward(&params);
            println!(
                "params: {:#?}, grads: {:#?}",
                params, grads_map, 
            );
            let new_params = optim.update(params, grads_map);
            println!("new_params: {:?}", new_params);
            params = new_params;
            println!("loss: {}", loss.item().unwrap());
            println!("\n-------------------\n");
            sleep(Duration::from_millis(50));
            if loss.item().unwrap() < 1e-6 {
                break;
            }
        }

        return params;
    }

    println!("Training model...");
    let params = train_model();
    println!("{:?}", params);
}