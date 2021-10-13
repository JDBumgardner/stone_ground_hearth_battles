#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum EventTypes {
    MonsterSummon {
        card:&MonsterCard
    },
    MonsterDeath
}