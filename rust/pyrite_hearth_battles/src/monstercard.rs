use std::fmt::Debug;
use std::fmt;
use super::monstercards::*;
use super::stattype::Stat;

#[derive(Clone, Eq, PartialEq)]
pub struct MonsterCard {
    pub card: MonsterCards,
    pub health: Stat,
    pub attack: Stat,
    pub tavern_tier: i8,
    pub pacifist: bool,
}

impl Debug for MonsterCard {
    fn fmt (&self, f:&mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, " stats: {}/{}, tavern tier: {}", self.attack, self.health, self.tavern_tier)
    }
}

impl MonsterCard {
    pub fn new(card:MonsterCards) -> MonsterCard {
        let stats: BaseProperties = card.get_base_stats();
        return MonsterCard { card: card, health: stats.health, attack: stats.attack, tavern_tier: stats.tier, pacifist: stats.pacifist }
    }
    pub fn cant_attack(&self) -> bool {
        self.pacifist || self.attack <= 0
    }
}
