use std::fmt::Debug;
use std::fmt;
use super::monstercards::*;

#[derive(Clone, Eq, PartialEq)]
pub struct MonsterCard {
    pub card: MonsterCards,
    pub properties: BaseProperties,
}

impl Debug for MonsterCard {
    fn fmt (&self, f:&mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, " stats: {}/{}, tavern tier: {}", self.properties.attack, self.properties.health, self.properties.tier)
    }
}

impl MonsterCard {
    pub fn new(card:MonsterCards) -> MonsterCard {
        return MonsterCard { card: card, properties: card.get_base_stats() }
    }
    pub fn cant_attack(&self) -> bool {
        self.properties.pacifist || self.properties.attack <= 0
    }
}
