use super::stattype::*;
use super::monstertypes::*;

#[derive(Clone, Eq, PartialEq)]
pub enum MonsterCards {
    AlleyCat,
    VulgarHomunculus,
    RabidSaurolisk,
}


pub struct BaseProperties{
    pub health: Stat,
    pub attack: Stat,
    pub tier: Tier,
    pub monstertype: MonsterTypes,
    pub pacifist: bool
}

impl MonsterCards {
    pub fn get_base_stats(&self) -> BaseProperties {
        match self {
            AlleyCat => BaseProperties{health: 1, attack: 1, tier: 1, monstertype: MonsterTypes::Beast, pacifist: false},
            VulgarHomunculus => BaseProperties{health: 4, attack: 2, tier: 1, monstertype: MonsterTypes::Demon, pacifist: false},
            RabidSaurolisk => BaseProperties{health: 2, attack: 3, tier: 1, monstertype: MonsterTypes::Beast, pacifist: false},
        }
    }
}
