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
    let card = MonsterCard::new(MonsterCards::VulgarHomunculus);
    assert_eq!(card.properties.health, 4);
    assert_eq!(card.properties.attack, 2);
}

#[test]
fn test_fruitless_war() {
    init();
    let mut warparty1 = WarParty::new(vec![MonsterCard{card: MonsterCards::AlleyCat, properties: BaseProperties::new(20, 2, 2, MonsterTypes::Beast)}]);
    let mut warparty2 = WarParty::new(vec![MonsterCard{card: MonsterCards::AlleyCat, properties: BaseProperties::new(4, 3, 2, MonsterTypes::Beast)}]);
    combat::battle_boards(&mut warparty1, &mut warparty2);
    assert_eq!(warparty1[0], MonsterCard{card: MonsterCards::AlleyCat, properties: BaseProperties::new(14, 2, 2, MonsterTypes::Beast)});
    assert_eq!(warparty2.len(), 0);
    println!("{:?} {:?}", warparty1, warparty2)
}

#[test]
fn test_taunt() {
    init();
    let mut warparty1 = WarParty::new(vec![MonsterCard::new(MonsterCards::VulgarHomunculus), MonsterCard::new(MonsterCards::AlleyCat), MonsterCard::new(MonsterCards::AlleyCat)]);
    let mut warparty2 = WarParty::new(vec![MonsterCard::new(MonsterCards::VulgarHomunculus), MonsterCard::new(MonsterCards::RabidSaurolisk)]);
    combat::battle_boards(&mut warparty1, &mut warparty2);
    assert_eq!(warparty1.len(), 0);
    assert_eq!(warparty2.len(), 0);
    println!("{:?} {:?}", warparty1, warparty2)
}