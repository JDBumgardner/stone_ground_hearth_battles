pub mod warparty;
pub mod monstercard;
pub mod monstercards;
pub mod combat;
pub mod stattype;
pub mod monstertypes;
use rand::{Rng};
use std::{
    option::Option,
    usize,
};
use warparty::WarParty;
use monstercard::MonsterCard;
use monstercards::*;



fn main() {}

struct Hand {
    cards: Vec<MonsterCard>,
}