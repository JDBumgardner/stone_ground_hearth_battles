use std::{cell::RefCell, rc::Rc};

use crate::monstercard::MonsterCard;

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum EventTypes {
    MonsterSummon {
        card:Rc<RefCell<MonsterCard>>
    },
    MonsterDeath {
        card:Rc<RefCell<MonsterCard>>
    }
}