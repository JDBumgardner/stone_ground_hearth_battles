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
    pub fn get_next_attacker_index(&mut self) -> Option<usize> {
        if !self.attacker_died {
            self.iterate_attacker_index();
        }
        if self.cards.is_empty() {
            return None
        }
        let original_index = self.attacker_index;
        loop {
            if self.cards[self.attacker_index].cant_attack() {
                self.iterate_attacker_index();
            } else {
                break
            }
            if self.attacker_index == original_index {
                return None
            }
        }
        self.attacker_died = false;
        return Some(self.attacker_index);
    }
    pub fn has_attacker(&self) -> bool {
        self.cards.iter().any(|x| !x.cant_attack())
    }
    pub fn iterate_attacker_index(&mut self) {
        self.attacker_index += 1;
        if self.attacker_index == self.cards.len() {
            self.attacker_index = 0;
        }
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
