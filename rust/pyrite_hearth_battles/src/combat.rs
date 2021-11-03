
use rand::{Rng};
use std::{
    option::Option,
    usize,
};
use super::warparty::WarParty;
use super::monstercard::MonsterCard;
use rand::prelude::SliceRandom;

pub fn battle_boards<'a>(attacker: &'a mut WarParty, defender: &'a mut WarParty) {
    let player_two_active: bool = rand::random();
    if player_two_active {
        std::mem::swap(attacker, defender)
    }
    loop {
        match (attacker.get_next_attacker_index(), select_target(defender)) {
            (Some(attacker_index), Some(defender_index)) => {
                println!("the attacker is: {:?}", attacker );
                println!("the defender is: {:?}", defender );
                fight(&mut attacker.index_mut(attacker_index), &mut defender.index_mut(defender_index));
                check_casualties(attacker, defender);
            }
            (None, Some(_)) => {
                if !defender.has_attacker() {
                    break
                }
            }
            _ => break,
        }
        std::mem::swap(attacker, defender);
    }
}

fn fight(attacker: &mut MonsterCard, defender: &mut MonsterCard) {
    attacker.properties.health -= defender.properties.attack;
    defender.properties.health -= attacker.properties.attack;
}

fn check_casualties(attacker_party: &mut WarParty, defender_party: &mut WarParty) {
    let mut card_index: usize = 0;
    while card_index < attacker_party.len() {
        if attacker_party.index_mut(card_index).properties.health <= 0 {
            attacker_party.remove(card_index)
        } else {
            card_index += 1;
        }
    }
    card_index = 0;
    while card_index < defender_party.len() {
        if defender_party.index_mut(card_index).properties.health <= 0 {
            defender_party.remove(card_index)
        } else {
            card_index += 1
        }
    }
}

fn select_target(defender: &WarParty) -> Option<usize> {
    let taunt_indices: Vec<usize> = defender.iter().enumerate().filter_map(
        |(index, monstercard)| if monstercard.properties.taunt { Some(index) } else { None }
    ).collect();
    if taunt_indices.len() > 0 {
       taunt_indices.choose(&mut rand::thread_rng()).map(
           |&x| x
       )
    } else if defender.len() > 0 {
        Some(rand::thread_rng().gen_range(0..defender.len()))
    } else {
        None
    }
}
