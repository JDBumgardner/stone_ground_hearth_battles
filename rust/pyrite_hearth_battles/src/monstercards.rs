use super::stattype::*;
use super::monstertypes::*;

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct BaseProperties{
    pub attack: Stat,
    pub health: Stat,
    pub tier: Tier,
    pub monstertype: MonsterTypes,
    pub pacifist: bool,
    pub taunt: bool
}

impl BaseProperties {
    pub fn new(attack: Stat, health: Stat, tier: Tier, monstertype: MonsterTypes) -> BaseProperties {
        return BaseProperties {
            attack: attack,
            health: health,
            tier: tier,
            monstertype: monstertype,
            pacifist: false,
            taunt: false
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum MonsterName {
    AlleyCat,
    VulgarHomunculus,
    RabidSaurolisk,
    ScavengingHyena,
    DragonSpawnLieutenant
}

impl MonsterName {
    pub fn get_base_stats(&self) -> BaseProperties {
        match self {            
            MonsterName::AlleyCat => BaseProperties::new(1, 1, 1, MonsterTypes::Beast),
            MonsterName::VulgarHomunculus => BaseProperties{taunt: true, ..BaseProperties::new( 2, 4, 1, MonsterTypes::Demon)},
            MonsterName::RabidSaurolisk => BaseProperties::new(3, 2, 1, MonsterTypes::Beast),
            MonsterName::ScavengingHyena => BaseProperties::new(2, 2, 1, MonsterTypes::Beast),
            MonsterName::DragonSpawnLieutenant => BaseProperties{taunt: true, ..BaseProperties::new(2, 3,  1, MonsterTypes::Dragon)}
        }
    }
}
