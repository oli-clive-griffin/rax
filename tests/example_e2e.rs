use rusty_grad::backward::{accum_grads, GradMap};
use rusty_grad::grad;
use rusty_grad::node::Node;
use rusty_grad::ops::{add, mmul, relu, sqr, sub};
use rusty_grad::optimizer::{Optimizer, ParamsMap, SGD};
use rusty_grad::tensor::Tensor;
use std::collections::HashMap;
use std::rc::Rc;

#[test]
fn test_train_mlp() {
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
        let x2 = relu(add(mmul(x1, w2), b2));
        let x3 = relu(add(mmul(x2, w3), b3));

        x3
    }

    fn forward(params: &ParamsMap, input: Tensor, label: Tensor) -> Rc<Node> {
        let label = Rc::new(Node::TensorParam(label, "label"));
        let out = model(params, input);

        sqr(sub(out.clone(), label.clone()))
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

        for i in 0.. {
            let (loss, grads_map) = grad!(forward, &params, x.clone(), y.clone());
            params = optim.update(params, grads_map);

            if i % 100 == 0 {
                println!("loss: {:.4}", loss.item().unwrap());
            }

            if loss.item().unwrap() < 1e-6 {
                break;
            }

        }

        params
    }

    println!("Training model...");
    let params = train_model();
    println!("{:?}", params);
}

#[test]
fn test_train_simple() {
    fn model(params: &ParamsMap) -> Rc<Node> {
        let a = Rc::new(Node::TensorParam(params.0.get("a").unwrap().clone(), "a"));
        let b = Rc::new(Node::TensorParam(params.0.get("b").unwrap().clone(), "b"));

        add(a, b)
    }

    fn train_model() -> ParamsMap {
        let mut params = ParamsMap(HashMap::from([
            ("a".to_string(), Tensor::rand(&vec![1])),
            ("b".to_string(), Tensor::rand(&vec![1])),
        ]));

        let optim = SGD::default();

        for i in 0.. {
            let (loss, grads_map) = grad!(model, &params);

            params = optim.update(params, grads_map);

            if i % 10 == 0 {
                println!("loss: {:.4}", loss.item().unwrap());
            }

            if loss.item().unwrap() < 1e-6 {
                break;
            }
        }

        params
    }

    println!("Training model...");
    let params = train_model();
    println!("{:?}", params);
}
