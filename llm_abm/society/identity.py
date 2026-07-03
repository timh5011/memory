"""Identity sampling: who the people of Polis are.

An identity is everything fixed at birth: name, occupation, class of origin,
temperament, household, friendships, faction, and — most importantly — the
agent's VALUE WEIGHTS: a normalized vector over the four fulfillment
dimensions saying what this person actually cares about. Someone with
belonging=0.55 will (and should) make different choices than someone with
prosperity=0.55. The weights go verbatim into the system prompt and into the
agent's own fulfillment metric F_i = Σ_d w_d · state_d.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DIMENSIONS = ("prosperity", "belonging", "standing", "security")

NAMES = [
    "Mara", "Josef", "Ilya", "Wren", "Tomas", "Petra", "Anselm", "Livia",
    "Bram", "Odile", "Caspar", "Noor", "Edda", "Rafael", "Sena", "Viktor",
    "Amara", "Dario", "Yuki", "Helene", "Owen", "Zaia", "Milan", "Freya",
    "Iris", "Silas", "Nadia", "Emil", "Thea", "Ruben", "Alba", "Corin",
]

# role: (base wage, wage variance factor, starting standing)
ROLES = {
    "farmer":   (4.0, 0.15, 0.5),
    "laborer":  (3.5, 0.10, 0.5),
    "artisan":  (5.0, 0.20, 0.8),
    "teacher":  (4.5, 0.05, 1.2),
    "healer":   (5.0, 0.10, 1.5),
    "merchant": (6.0, 0.40, 0.8),
    "official": (6.5, 0.05, 2.0),
    "priest":   (4.0, 0.05, 2.0),
}

CLASS_ROLES = {
    "lower":  ["farmer", "laborer", "farmer", "laborer", "artisan"],
    "middle": ["artisan", "teacher", "healer", "merchant", "priest", "farmer"],
    "upper":  ["merchant", "official", "official", "healer", "priest"],
}

CLASS_WEALTH = {"lower": 10.0, "middle": 25.0, "upper": 60.0}
CLASS_PROBS = {"lower": 0.4, "middle": 0.4, "upper": 0.2}

FACTIONS = ("Hearth League", "Meridian Society")  # communalist / individualist


@dataclass
class Identity:
    agent_id: int
    name: str
    role: str
    social_class: str            # class of origin (fixed; current rank can move)
    temperament: dict[str, float]  # ambition, gregariousness, progressivism ∈ [-1,1]
    household: list[int] = field(default_factory=list)  # ids incl. self
    friends: list[int] = field(default_factory=list)
    faction: str = FACTIONS[0]
    value_weights: dict[str, float] = field(default_factory=dict)

    @property
    def dominant_value(self) -> str:
        return max(self.value_weights, key=self.value_weights.get)

    def to_record(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "social_class": self.social_class,
            "temperament": self.temperament,
            "household": self.household,
            "friends": self.friends,
            "faction": self.faction,
            "value_weights": self.value_weights,
            "dominant_value": self.dominant_value,
        }


def _sample_value_weights(rng: np.random.Generator,
                          temperament: dict[str, float]) -> dict[str, float]:
    """Dirichlet weights over dimensions, biased by temperament.

    Ambition pulls toward prosperity/standing; gregariousness toward
    belonging; contentment toward security. The bias keeps personalities and
    values coherent without making them deterministic.
    """
    alpha = np.array([1.0, 1.0, 1.0, 1.0])
    alpha[0] += max(temperament["ambition"], 0) * 1.5        # prosperity
    alpha[1] += max(temperament["gregariousness"], 0) * 1.5  # belonging
    alpha[2] += max(temperament["ambition"], 0) * 1.0        # standing
    alpha[3] += max(-temperament["ambition"], 0) * 1.5       # security
    w = rng.dirichlet(alpha)
    return {d: round(float(x), 3) for d, x in zip(DIMENSIONS, w)}


def sample_population(n_agents: int, rng: np.random.Generator) -> list[Identity]:
    """Sample a full population: identities, households, friendships, factions."""
    assert n_agents <= len(NAMES), "add more names for larger populations"
    identities: list[Identity] = []

    classes = rng.choice(list(CLASS_PROBS), size=n_agents,
                         p=list(CLASS_PROBS.values()))
    for i in range(n_agents):
        social_class = str(classes[i])
        temperament = {
            "ambition": round(float(rng.uniform(-1, 1)), 2),
            "gregariousness": round(float(rng.uniform(-1, 1)), 2),
            "progressivism": round(float(rng.uniform(-1, 1)), 2),
        }
        role = str(rng.choice(CLASS_ROLES[social_class]))
        # Progressives lean individualist (Meridian), traditionals communalist
        p_meridian = 0.5 + 0.35 * temperament["progressivism"]
        faction = FACTIONS[1] if rng.random() < p_meridian else FACTIONS[0]
        identities.append(Identity(
            agent_id=i,
            name=NAMES[i],
            role=role,
            social_class=social_class,
            temperament=temperament,
            faction=faction,
            value_weights=_sample_value_weights(rng, temperament),
        ))

    # Households: partition the population into groups of 2-4
    ids = list(rng.permutation(n_agents))
    while ids:
        size = int(rng.integers(2, 5))
        members = [int(x) for x in ids[:size]]
        ids = ids[size:]
        if len(ids) == 1:  # avoid a lonely leftover household
            members.append(int(ids.pop()))
        for m in members:
            identities[m].household = members

    # Friendships: 2-4 ties outside the household
    for ident in identities:
        n_friends = int(rng.integers(2, 5))
        candidates = [j for j in range(n_agents)
                      if j != ident.agent_id and j not in ident.household]
        friends = rng.choice(candidates, size=min(n_friends, len(candidates)),
                             replace=False)
        ident.friends = sorted(set(int(f) for f in friends))

    return identities
