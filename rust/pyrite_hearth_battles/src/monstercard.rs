use std::fmt::Debug;
use std::fmt;

use super::monstercards::*;
use super::eventtypes::*;
use super::monstertypes::*;

#[derive(Clone, Eq, PartialEq)]
pub struct MonsterCard {
    pub card_name: MonsterName,
    pub properties: BaseProperties,
}

impl Debug for MonsterCard {
    fn fmt (&self, f:&mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, " stats: {}/{}, tavern tier: {}", self.properties.attack, self.properties.health, self.properties.tier)
    }
}

impl MonsterCard {
    pub fn new(card:MonsterName) -> MonsterCard {
        return MonsterCard { card_name: card, properties: card.get_base_stats() }
    }
    pub fn cant_attack(&self) -> bool {
        self.properties.pacifist || self.properties.attack <= 0
    }

    pub fn event_handler(&mut self, event: &EventTypes) {
        match self.card_name {
            MonsterName::AlleyCat => { },
            MonsterName::RabidSaurolisk => { },
            MonsterName::ScavengingHyena => { 
                match event {
                   EventTypes::MonsterDeath { card } => {
                        if card.borrow().properties.monstertype == MonsterTypes::Beast {
                            self.properties.attack += 2; 
                            self.properties.health += 1; 
                        }
                    },
                    _ => { }
                }
            },
            MonsterName::VulgarHomunculus => { }
        }
    }
}
