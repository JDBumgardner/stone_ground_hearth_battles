use pyrite_hearth_battles::warparty::WarParty;
use pyrite_hearth_battles::monstercard::MonsterCard;
use pyrite_hearth_battles::combat;
use pyrite_hearth_battles::monstercards::*;

fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn test_fruitless_war() {
    init();
    let mut warparty1 = WarParty::new(vec![MonsterCard{card: MonsterCards::AlleyCat, health: 20, attack: 2, tavern_tier: 2, pacifist: false}]);
    let mut warparty2 = WarParty::new(vec![MonsterCard{card: MonsterCards::AlleyCat, health: 4, attack: 3, tavern_tier: 2, pacifist: false}]);
    combat::battle_boards(&mut warparty1, &mut warparty2);
    assert_eq!(warparty1[0], MonsterCard{card: MonsterCards::AlleyCat, health: 14, attack: 2, tavern_tier: 2, pacifist: false});
    assert_eq!(warparty2.len(), 0);
    println!("{:?} {:?}", warparty1, warparty2)

}