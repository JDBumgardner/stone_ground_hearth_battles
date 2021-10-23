use std::{cell::{Ref, RefCell, RefMut}, rc::Rc};

use crate::eventtypes::EventTypes;

use super::monstercard::MonsterCard;
#[derive(Clone, Debug)]
pub struct WarParty {
    attacker_index: usize,
    attacker_died: bool,
    cards: Vec<Rc<RefCell<MonsterCard>>>,
}

impl WarParty {
    pub fn new(cards: Vec<MonsterCard>) -> Self {
        WarParty {
            attacker_index: 0,
            attacker_died: true,
            cards: cards.into_iter().map(|x| Rc::new(RefCell::new(x))).collect(),
        }
    }
    pub fn insert(&mut self, position: usize, card: MonsterCard) {
        self.cards.insert(position, Rc::new(RefCell::new(card)));
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
            if self.index_mut(self.attacker_index).cant_attack() {
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
        self.iter().any(|x| !x.cant_attack())
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
    pub fn iter(&self) -> WarPartyIterator {
        return WarPartyIterator(self.cards.iter())
    }

    pub fn index(&self, index: usize) -> Ref<MonsterCard> {
        self.cards[index].borrow()
    }

    pub fn index_mut(&mut self, index: usize) -> RefMut<MonsterCard> {
        self.cards[index].borrow_mut()
    }

    pub fn broadcaast_event(&mut self, event: EventTypes)
    {
        for card in &self.cards {
            card.borrow_mut().event_handler(&event);
        }
    }
}

pub struct WarPartyIterator<'a>(std::slice::Iter<'a,Rc<RefCell<MonsterCard>>>);

impl<'a> Iterator for WarPartyIterator<'a> {
    type Item = Ref<'a, MonsterCard>;
    
    fn next(&mut self) -> Option<Self::Item> {
        match self.0.next() {
            None => None,
            Some(rc) => Some(rc.borrow())
        }
    }
}