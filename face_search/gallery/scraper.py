"""Scraping da lista publica de procurados do Ministerio da Justica.

A lista de procurados do MJSP e publica, paginada e contem, para cada pessoa,
um link para a pagina de detalhe e uma foto oficial. Este modulo percorre as
paginas, extrai os dados de cada procurado e localiza a URL da foto.

Nenhuma imagem e gravada aqui — o scraper so descobre os registros e suas
URLs. O download e a codificacao ficam a cargo do ``builder``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Pagina inicial da lista publica de procurados.
LIST_URL = (
    "https://www.gov.br/mj/pt-br/assuntos/sua-seguranca/seguranca-publica/"
    "operacoes-integradas/projeto-captura/lista-de-procurados"
)

# User-Agent identificavel, declarando a finalidade academica do acesso.
USER_AGENT = "mj-procurados-sync/1.0 (+academic dataset sync)"

# Expressoes regulares usadas para interpretar os textos da pagina.
DATE_RE = re.compile(r"(?P<date>\d{2}/\d{2}/\d{4})")
UPDATED_RE = re.compile(r"Atualizado em\s+(\d{2}/\d{2}/\d{4}\s+\d{2}h\d{2})")
LIST_ITEM_RE = re.compile(r"^(?P<title>.+?)\s+(?P<date>\d{2}/\d{2}/\d{4})$")


@dataclass(frozen=True)
class ListedPerson:
    """Um procurado descoberto na listagem publica."""

    record_id: str          # identificador estavel, derivado da URL
    name: str               # nome da pessoa
    state: str              # UF associada ao registro (pode ser vazia)
    listed_date: str | None # data em que o registro foi listado
    source_url: str         # URL da pagina de detalhe


def normalize_space(value: str) -> str:
    """Colapsa espacos em branco repetidos e remove as bordas."""

    return re.sub(r"\s+", " ", value).strip()


def build_session() -> requests.Session:
    """Cria uma sessao HTTP com retentativa automatica.

    Reaproveitar a sessao mantem a conexao viva entre paginas. A politica de
    retry cobre erros transitorios (429, 5xx) com backoff exponencial.
    """

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch(session: requests.Session, url: str) -> requests.Response:
    """Faz um GET e levanta excecao se o status nao for de sucesso."""

    response = session.get(url, timeout=45)
    response.raise_for_status()
    return response


def record_id_from_url(url: str) -> str:
    """Extrai um identificador estavel a partir da URL de detalhe.

    O ultimo segmento do caminho (sem o sufixo ``/view``) e unico e estavel,
    entao serve de chave do registro no manifesto.
    """

    path = urlparse(url).path.rstrip("/")
    if path.endswith("/view"):
        path = path[: -len("/view")]
    return path.rsplit("/", 1)[-1]


def parse_name_state(raw_title: str) -> tuple[str, str]:
    """Separa o titulo de um item em nome da pessoa e UF.

    O titulo costuma vir no formato ``"NOME DA PESSOA - UF"``. Quando nao ha
    UF reconhecivel, devolve a UF vazia.
    """

    title = normalize_space(DATE_RE.sub("", raw_title))
    match = re.match(
        r"^(?P<name>.+?)\s*-\s*(?P<state>[A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ ]+)\.?$", title
    )
    if not match:
        return title.rstrip(".").strip(), ""

    name = match.group("name").strip().rstrip(".")
    state = match.group("state").strip().rstrip(".")
    return name, state


def parse_listing(html: str, page_url: str) -> tuple[list[ListedPerson], str | None]:
    """Extrai os procurados de uma pagina da listagem.

    Devolve a lista de pessoas encontradas e a URL da proxima pagina (ou
    ``None`` se esta for a ultima). Registros repetidos na mesma pagina sao
    ignorados.
    """

    soup = BeautifulSoup(html, "html.parser")
    records: list[ListedPerson] = []
    seen: set[str] = set()
    next_url: str | None = None

    for anchor in soup.find_all("a", href=True):
        text = normalize_space(anchor.get_text(" "))
        href = urljoin(page_url, anchor["href"])

        # Link de paginacao "Proximo".
        if "Próximo" in text:
            next_url = href
            continue

        # So nos interessam links de detalhe de procurado.
        if "/lista-de-procurados/" not in href or not href.rstrip("/").endswith(
            "/view"
        ):
            continue

        match = LIST_ITEM_RE.match(text)
        if not match:
            continue

        source_url = href
        record_id = record_id_from_url(source_url)
        if record_id in seen:
            continue

        name, state = parse_name_state(match.group("title"))
        records.append(
            ListedPerson(
                record_id=record_id,
                name=name,
                state=state,
                listed_date=match.group("date"),
                source_url=source_url,
            )
        )
        seen.add(record_id)

    return records, next_url


def image_url_for(source_url: str) -> str:
    """Monta a URL da foto oficial a partir da URL de detalhe."""

    path = source_url.rstrip("/")
    if path.endswith("/view"):
        path = path[: -len("/view")]
    return f"{path}/@@images/image"


def parse_updated_at(html: str) -> str | None:
    """Le a data de "Atualizado em" da pagina de detalhe.

    Essa data e usada como chave de cache: se ela nao mudou, o embedding ja
    gerado pode ser reaproveitado sem baixar a foto de novo.
    """

    text = normalize_space(BeautifulSoup(html, "html.parser").get_text(" "))
    match = UPDATED_RE.search(text)
    return match.group(1) if match else None


def collect_listed_people(
    session: requests.Session,
    limit: int | None,
    delay_seconds: float,
) -> list[ListedPerson]:
    """Percorre todas as paginas da listagem e devolve os procurados.

    A varredura para quando nao ha proxima pagina, quando uma pagina se repete
    ou quando o ``limit`` opcional e atingido. Entre paginas ha uma pausa
    (``delay_seconds``), para nao sobrecarregar o site publico.
    """

    page_url: str | None = LIST_URL
    people: list[ListedPerson] = []
    seen_pages: set[str] = set()
    seen_records: set[str] = set()

    while page_url and page_url not in seen_pages:
        seen_pages.add(page_url)
        response = fetch(session, page_url)
        page_people, next_url = parse_listing(response.text, page_url)

        new_people = [
            person for person in page_people if person.record_id not in seen_records
        ]
        people.extend(new_people)
        seen_records.update(person.record_id for person in new_people)
        print(f"coletados {len(new_people)} registros de {page_url}")

        if limit and len(people) >= limit:
            return people[:limit]

        page_url = next_url
        if page_url:
            time.sleep(delay_seconds)

    return people
