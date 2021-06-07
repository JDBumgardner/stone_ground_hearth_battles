use pyrite_hearth_battles::warparty::WarParty;
use pyrite_hearth_battles::monstercard::MonsterCard;
use pyrite_hearth_battles::combat;



#[test]
fn test_fruitless_war() {
    let mut warparty1 = WarParty::new(vec![MonsterCard{health: 20, attack: 2, tavern_tier: 2}]);
    let mut warparty2 = WarParty::new(vec![MonsterCard{health: 4, attack: 3, tavern_tier: 2}]);
    combat::battle_boards(&mut warparty1, &mut warparty2);
    println!("{:?} {:?}", warparty1, warparty2)

}