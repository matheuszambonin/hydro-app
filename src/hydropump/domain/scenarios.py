"""Cenários hidrogeológicos e a metodologia de interpretação de cada um.

Motivação
---------
"Rodar Cooper-Jacob" não é uma decisão única: a mesma reta semilog significa
coisas diferentes conforme o sistema. Este módulo torna explícito **qual
cenário o usuário declarou** e, a partir disso, define:

- se a correção de Jacob para aquífero livre se aplica por padrão;
- qual faixa de ``S`` é fisicamente plausível (produção específica vs.
  armazenamento elástico vs. porosidade de fraturas);
- que assinaturas anômalas procurar nos dados (dupla porosidade, drenança);
- que ressalvas obrigatoriamente acompanham o resultado.

Honestidade sobre o alcance
---------------------------
Nenhum cenário aqui "resolve" a heterogeneidade. Cooper-Jacob continua sendo
um modelo de meio equivalente, homogêneo, isotrópico e de extensão infinita.
O que os cenários fazem é (a) ajustar defaults e faixas de validação, e
(b) **avisar** quando os dados exibem um comportamento que o modelo escolhido
não descreve. Estimar parâmetros de dupla porosidade de verdade (ω, λ do
modelo de Warren-Root/Moench) exige ajuste de curvas-tipo, que este módulo
deliberadamente **não** faz — ele apenas sinaliza a assinatura.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "AquiferScenario",
    "ScenarioProfile",
    "SCENARIO_PROFILES",
    "profile_for",
]


class AquiferScenario(StrEnum):
    """Cenário hidrogeológico declarado pelo usuário."""

    UNCONFINED_GRANULAR = "unconfined_granular"
    CONFINED = "confined"
    UNCONFINED_LATERITIC_BASE = "unconfined_lateritic_base"
    MIXED_GRANULAR_FRACTURED = "mixed_granular_fractured"

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELS: dict[AquiferScenario, str] = {
    AquiferScenario.UNCONFINED_GRANULAR: "Aquífero livre granular (genérico)",
    AquiferScenario.CONFINED: "Aquífero confinado / semiconfinado",
    AquiferScenario.UNCONFINED_LATERITIC_BASE: (
        "Livre granular, penetração total sobre base laterítica"
    ),
    AquiferScenario.MIXED_GRANULAR_FRACTURED: (
        "Misto: granular no topo, fraturado na base"
    ),
}


@dataclass(frozen=True, slots=True)
class ScenarioProfile:
    """Regras de interpretação associadas a um cenário.

    Attributes
    ----------
    scenario : O cenário.
    label : Nome legível.
    apply_jacob_by_default : Se a correção de Jacob deve vir marcada.
    unconfined : Se a faixa plausível de S é a de aquífero livre (S_y).
    storativity_range : Faixa plausível de S para este cenário.
    storativity_note : Como interpretar o S obtido.
    methodology : Descrição do método aplicado, para o relatório.
    caveats : Ressalvas que sempre acompanham este cenário.
    check_dual_porosity : Se vale procurar a assinatura de dupla porosidade.
    check_leakage : Se vale procurar achatamento tardio (drenança/recarga).
    b_is_well_constrained : Se a espessura saturada é conhecida com confiança
        (afeta o quanto confiar na correção de Jacob e em K = T/b).
    """

    scenario: AquiferScenario
    label: str
    apply_jacob_by_default: bool
    unconfined: bool
    storativity_range: tuple[float, float]
    storativity_note: str
    methodology: str
    caveats: tuple[str, ...]
    check_dual_porosity: bool = False
    check_leakage: bool = False
    b_is_well_constrained: bool = False


SCENARIO_PROFILES: dict[AquiferScenario, ScenarioProfile] = {
    AquiferScenario.UNCONFINED_GRANULAR: ScenarioProfile(
        scenario=AquiferScenario.UNCONFINED_GRANULAR,
        label=_LABELS[AquiferScenario.UNCONFINED_GRANULAR],
        apply_jacob_by_default=True,
        unconfined=True,
        storativity_range=(0.01, 0.35),
        storativity_note=(
            "S aproxima a produção específica (S_y) do material granular."
        ),
        methodology=(
            "Cooper-Jacob (aproximação logarítmica de Theis) com correção de "
            "Jacob (1944) para redução da espessura saturada."
        ),
        caveats=(
            "Assume aquífero homogêneo, isotrópico, de extensão infinita e "
            "vazão constante.",
            "A correção de Jacob é confiável apenas enquanto s/b < 25%.",
        ),
        check_leakage=True,
    ),
    AquiferScenario.CONFINED: ScenarioProfile(
        scenario=AquiferScenario.CONFINED,
        label=_LABELS[AquiferScenario.CONFINED],
        apply_jacob_by_default=False,
        unconfined=False,
        storativity_range=(1.0e-5, 1.0e-3),
        storativity_note=(
            "S é o armazenamento elástico (compressibilidade da água e do "
            "esqueleto sólido), ordens de grandeza menor que em aquífero livre."
        ),
        methodology=(
            "Cooper-Jacob (aproximação logarítmica de Theis), sem correção de "
            "Jacob — em aquífero confinado a espessura saturada não varia com "
            "o rebaixamento."
        ),
        caveats=(
            "Se houver drenança de camada adjacente (aquífero semiconfinado), "
            "a curva se achata no tempo tardio e Cooper-Jacob superestima T; "
            "nesse caso o método correto é Hantush-Jacob.",
        ),
        check_leakage=True,
    ),
    AquiferScenario.UNCONFINED_LATERITIC_BASE: ScenarioProfile(
        scenario=AquiferScenario.UNCONFINED_LATERITIC_BASE,
        label=_LABELS[AquiferScenario.UNCONFINED_LATERITIC_BASE],
        apply_jacob_by_default=True,
        unconfined=True,
        storativity_range=(0.01, 0.35),
        storativity_note=(
            "S aproxima a produção específica (S_y) do pacote granular acima "
            "da laterita."
        ),
        methodology=(
            "Cooper-Jacob com correção de Jacob (1944). Este é o cenário mais "
            "favorável ao método: a base laterítica define fisicamente a "
            "espessura saturada b, e a penetração total do poço no pacote "
            "granular valida a hipótese de fluxo radial — não há necessidade "
            "de correção por penetração parcial."
        ),
        caveats=(
            "Assume que a laterita se comporta como base impermeável. Se ela "
            "for apenas semipermeável, haverá drenança e a curva se achatará "
            "no tempo tardio — verifique o diagnóstico de achatamento.",
            "b bem definido torna K = T/b uma estimativa confiável, ao "
            "contrário do caso genérico.",
        ),
        check_leakage=True,
        b_is_well_constrained=True,
    ),
    AquiferScenario.MIXED_GRANULAR_FRACTURED: ScenarioProfile(
        scenario=AquiferScenario.MIXED_GRANULAR_FRACTURED,
        label=_LABELS[AquiferScenario.MIXED_GRANULAR_FRACTURED],
        apply_jacob_by_default=True,
        unconfined=True,
        # Faixa alargada: mistura porosidade de fraturas (muito baixa, no
        # tempo inicial) com produção específica do granular (tempo tardio).
        storativity_range=(1.0e-4, 0.35),
        storativity_note=(
            "Em sistema misto, S depende de QUAL trecho da curva foi ajustado: "
            "o trecho inicial reflete o armazenamento das fraturas (S baixo, "
            "1e-4 a 1e-2) e o trecho tardio, a produção específica do pacote "
            "granular (S_y, 0,01 a 0,35). Um valor intermediário costuma "
            "indicar que o ajuste cruzou a transição entre os dois."
        ),
        methodology=(
            "Cooper-Jacob aplicado como MEIO EQUIVALENTE: o T obtido é a "
            "transmissividade global do sistema (granular + fraturado "
            "conjugados), não a de nenhuma das porções isoladamente. Quando "
            "os dados exibem a assinatura de dupla porosidade (dois trechos "
            "retilíneos aproximadamente paralelos separados por um patamar), "
            "o software ajusta cada trecho separadamente e reporta os dois S."
        ),
        caveats=(
            "T é global. Separar a contribuição do granular e do fraturado "
            "exige dado adicional (perfilagem de fluxo, ensaio com obturador "
            "ou perfil geológico com ensaio por trecho) — não é recuperável "
            "de um único ensaio de bombeamento.",
            "A validade do meio equivalente depende de a densidade de "
            "fraturas ser alta o bastante para o meio se comportar como "
            "contínuo na escala do ensaio. Em rocha com poucas fraturas "
            "condutivas, o fluxo é canalizado e Cooper-Jacob não se aplica.",
            "Se a fronteira granular/fraturado for desconhecida, K = T/b "
            "é apenas um valor médio ponderado, sem significado para nenhuma "
            "das litologias em separado.",
        ),
        check_dual_porosity=True,
        check_leakage=True,
    ),
}


def profile_for(scenario: AquiferScenario) -> ScenarioProfile:
    """Perfil de interpretação do cenário.

    Examples
    --------
    >>> p = profile_for(AquiferScenario.UNCONFINED_LATERITIC_BASE)
    >>> p.b_is_well_constrained
    True
    >>> profile_for(AquiferScenario.CONFINED).apply_jacob_by_default
    False
    >>> profile_for(AquiferScenario.MIXED_GRANULAR_FRACTURED).check_dual_porosity
    True
    """
    return SCENARIO_PROFILES[scenario]
