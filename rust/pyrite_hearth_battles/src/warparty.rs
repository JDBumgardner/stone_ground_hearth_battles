use std::ops::{Index, IndexMut};

use super::monstercard::MonsterCard;
#[derive(Clone, Debug)]
pub struct WarParty {
    attacker_index: usize,
    attacker_died: bool,
    cards: Vec<MonsterCard>,
}

impl WarParty {
    pub fn new(cards: Vec<MonsterCard>) -> Self {
        WarParty {
            attacker_index: 0,
            attacker_died: true,
            cards: cards,
        }
    }
    pub fn insert(&mut self, position: usize, card: MonsterCard) {
        self.cards.insert(position, card);
        if self.attacker_index >= position {
            self.attacker_index += 1;
        }
    }
    pub fn remove(&mut self, position: usize) {
        self.cards.remove(position);
        if self.attacker_index > position {
            self.attacker_index -= 1;
        }
        if self.attacker_index == position {
            self.attacker_died = true;
        }
    }
    pub fn get_next_attacker_index(&mut self) -> usize {
        if !self.attacker_died {
            self.attacker_index += 1;
            // TODO: Wrap index around and deal with 0
        }
        self.attacker_died = false;
        return self.attacker_index;
    }
    pub fn is_empty(&self) -> bool {
        self.cards.is_empty()
    }
    pub fn len(&self) -> usize {
        self.cards.len()
    }
}

impl Index<usize> for WarParty {
    type Output = MonsterCard;

    fn index(&self, index: usize) -> &MonsterCard {
        &self.cards[index]
    }
}
impl IndexMut<usize> for WarParty {
    fn index_mut(&mut self, index: usize) -> &mut MonsterCard {
        &mut self.cards[index]
    }
}
