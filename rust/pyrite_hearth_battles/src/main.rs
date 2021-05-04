use rand::thread_rng;

fn main() {

}
#[derive(Clone)]
struct MonsterCard {
    health: i32,
    attack: i32,
    tavern_tier: i8,
}
#[derive(Clone)]
struct WarParty {
    attacker_index: usize,
    attacker_died: bool, 
    cards: Vec<MonsterCard>,
}

impl WarParty {
    fn new(cards: Vec<MonsterCard>) -> Self {
        WarParty{
            attacker_index: 0,
            attacker_died: true,
            cards: cards,
        }
    }
    fn insert (&mut self, position: usize, card: MonsterCard) {
        self.cards.insert(position, card);
        if self.attacker_index >= position {
            self.attacker_index += 1;
        }
    }
    fn remove (&mut self, position: usize) {
        self.cards.remove(position);
        if self.attacker_index > position {
            self.attacker_index -= 1;
        }
        if self.attacker_index == position{
            self.attacker_died == true;
        }
    }
    fn get_next_fight_index(&mut self) -> usize {
        if !self.attacker_died{
            self.attacker_index += 1;
            self.attacker_died = false;
        }
        return self.attacker_index;
    }
}
struct Hand {
    cards: Vec<MonsterCard>,
}

fn battle_boards(attacker: &mut WarParty, defender: &mut WarParty) {
    let player_two_active: bool = rand::random();
    let mut target_index: i8 = 0;
    if player_two_active {
        std::mem::swap(&mut attacker, &mut defender)
    } 
    while !end_condition {
        target_index = select_target(defender);
        
        active_player attacks not_active_player
        resolve_attack_stuff
        switch_active_player

        std::mem::swap(&mut attacker, &mut defender)
    }

}

fn fight(attacker: &mut MonsterCard, defender: &mut MonsterCard) {
    attacker.health -= defender.attack;
    defender.health -= attacker.attack;
}

fn select_target(defender: &WarParty) {
    let target: i8 = rand::random()
}

fn assemble(posse_1: Vec<MonsterCard>, posse_2: Vec<MonsterCard>) {
    return battle_boards(&mut WarParty::new(posse_1.clone()), &mut WarParty::new(posse_2.clone()))
}