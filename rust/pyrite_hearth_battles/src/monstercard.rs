use std::fmt::Debug;
use std::fmt;

#[derive(Clone)]
pub struct MonsterCard {
    pub health: i32,
    pub attack: i32,
    pub tavern_tier: i8,
    pub pacifist: bool,
}

impl Debug for MonsterCard {
    fn fmt (&self, f:&mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, " stats: {}/{}, tavern tier: {}", self.attack, self.health, self.tavern_tier)
    }
}

impl MonsterCard {
    pub fn cant_attack(&self) -> bool {
        self.pacifist || self.attack <= 0
    }
}
