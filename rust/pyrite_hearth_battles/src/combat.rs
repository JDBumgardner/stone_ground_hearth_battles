
use rand::{Rng};
use std::{
    option::Option,
    usize,
};
use super::warparty::WarParty;
use super::monstercard::MonsterCard;


pub fn battle_boards<'a>(attacker: &'a mut WarParty, defender: &'a mut WarParty) {
    let player_two_active: bool = rand::random();
    if player_two_active {
        std::mem::swap(attacker, defender)
    }
    loop {
        match select_target(defender) {
            Some(defender_index) => {
                let attacker_index: usize = attacker.get_next_attacker_index();
                fight(&mut attacker[attacker_index], &mut defender[defender_index]);
                check_casualties(attacker, defender);
                std::mem::swap(attacker, defender);
            }
            None => break,
        }
    }
}

fn fight(attacker: &mut MonsterCard, defender: &mut MonsterCard) {
    attacker.health -= defender.attack;
    defender.health -= attacker.attack;
}

fn check_casualties(attacker_party: &mut WarParty, defender_party: &mut WarParty) {
    let mut card_index: usize = 0;
    while card_index < attacker_party.len() {
        if attacker_party[card_index].health <= 0 {
            attacker_party.remove(card_index)
        } else {
            card_index += 1;
        }
    }
    card_index = 0;
    while card_index < defender_party.len() {
        if defender_party[card_index].health <= 0 {
            defender_party.remove(card_index)
        } else {
            card_index += 1
        }
    }
}

fn select_target(defender: &WarParty) -> Option<usize> {
    if defender.len() > 0 {
        Some(rand::thread_rng().gen_range(0..defender.len()))
    } else {
        None
    }
}

fn assemble(posse_1: Vec<MonsterCard>, posse_2: Vec<MonsterCard>) {
    return battle_boards(
        &mut WarParty::new(posse_1.clone()),
        &mut WarParty::new(posse_2.clone()),
    );
}
