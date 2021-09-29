use pyrite_hearth_battles::warparty::WarParty;
use pyrite_hearth_battles::monstercard::MonsterCard;
use pyrite_hearth_battles::combat;

fn init() {
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn test_fruitless_war() {
    init();
    let mut warparty1 = WarParty::new(vec![MonsterCard{health: 20, attack: 2, tavern_tier: 2, pacifist: false}]);
    let mut warparty2 = WarParty::new(vec![MonsterCard{health: 4, attack: 3, tavern_tier: 2, pacifist: false}]);
    combat::battle_boards(&mut warparty1, &mut warparty2);
    println!("{:?} {:?}", warparty1, warparty2)

}