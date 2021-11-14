use pyrite_hearth_battles::warparty::WarParty;
use pyrite_hearth_battles::monstercard::MonsterCard;
use pyrite_hearth_battles::combat;
use pyrite_hearth_battles::monstercards::*;
use pyrite_hearth_battles::monstertypes::*;

fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn test_vulgar_homunculus() {
    let card = MonsterCard::new(MonsterName::VulgarHomunculus);
    assert_eq!(card.properties.health, 4);
    assert_eq!(card.properties.attack, 2);
}

#[test]
fn test_fruitless_war() {
    init();
    let mut warparty1 = WarParty::new(vec![MonsterCard{card_name: MonsterName::AlleyCat, properties: BaseProperties::new(20, 2, 2, MonsterTypes::Beast)}]);
    let mut warparty2 = WarParty::new(vec![MonsterCard{card_name: MonsterName::AlleyCat, properties: BaseProperties::new(4, 3, 2, MonsterTypes::Beast)}]);
    combat::battle_boards(&mut warparty1, &mut warparty2);
    assert_eq!(*warparty1.index(0), MonsterCard{card_name: MonsterName::AlleyCat, properties: BaseProperties::new(14, 2, 2, MonsterTypes::Beast)});
    assert_eq!(warparty2.len(), 0);
    println!("{:?} {:?}", warparty1, warparty2)
}

#[test]
fn test_taunt() {
    init();
    let mut warparty1 = WarParty::new(vec![MonsterCard::new(MonsterName::VulgarHomunculus), MonsterCard::new(MonsterName::AlleyCat), MonsterCard::new(MonsterName::AlleyCat)]);
    let mut warparty2 = WarParty::new(vec![MonsterCard::new(MonsterName::VulgarHomunculus), MonsterCard::new(MonsterName::RabidSaurolisk)]);
    combat::battle_boards(&mut warparty1, &mut warparty2);
    assert_eq!(warparty1.len(), 0);
    assert_eq!(warparty2.len(), 0);
    println!("{:?} {:?}", warparty1, warparty2)
}

#[test]
fn test_scavenging_hyena() {
    init();
    let mut warparty1 = WarParty::new(vec![MonsterCard::new(MonsterName::DragonSpawnLieutenant), MonsterCard::new(MonsterName::DragonSpawnLieutenant)]);
    let mut warparty2 = WarParty::new(vec![MonsterCard::new(MonsterName::ScavengingHyena), MonsterCard::new(MonsterName::ScavengingHyena)]);
    combat::battle_boards(&mut warparty1, &mut warparty2);
    assert_eq!(warparty1.len(), 0);
    assert_eq!(warparty2.len(), 0);
    println!("{:?} {:?}", warparty1, warparty2)
}