pub mod warparty;
pub mod monstercard;
pub mod combat;
use rand::{Rng};
use std::{
    option::Option,
    usize,
};
use warparty::WarParty;
use monstercard::MonsterCard;



fn main() {}

struct Hand {
    cards: Vec<MonsterCard>,
}