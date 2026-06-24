import logging
from typing import Any, Callable, Optional
from config import get_settings

from domain.repositories.paper_repository import PaperRepository
from domain.repositories.patent_repository import PatentRepository
from domain.repositories.project_repository import ProjectRepository
from domain.repositories.researcher_repository import ResearcherRepository
from domain.repositories.technology_repository import TechnologyRepository
logger = logging.getLogger(__name__)

class RepositoryFactory:
    """
    도메인 레포지토리 객체를 생성하고 캐싱(싱글톤)하는 팩토리 클래스.
    """
    def __init__(self):
        settings = get_settings()
        self.mariadb_url = settings.mariadb_url
        self.is_mariadb = bool(self.mariadb_url)

        self.db_pool = None
        if self.is_mariadb:
            from infrastructure.database import PyMySQLPool
            self.db_pool = PyMySQLPool(self.mariadb_url)

        self._paper_repo: Optional[PaperRepository] = None
        self._patent_repo: Optional[PatentRepository] = None
        self._project_repo: Optional[ProjectRepository] = None
        self._researcher_repo: Optional[ResearcherRepository] = None
        self._technology_repo: Optional[TechnologyRepository] = None

        # Milvus / Neo4j (lazy init)
        self._vector_search_fn: Optional[Callable] = None
        self._neo4j_driver: Any = None
        self._graph_query_fn: Optional[Callable] = None
        self._researcher_network_fn: Optional[Callable] = None
        self._citation_graph_fn: Optional[Callable] = None

    def get_paper_repository(self) -> PaperRepository:
        if self._paper_repo is None:
            if self.is_mariadb:
                from infrastructure.repositories.paper_repository import MariaDBPaperRepository
                self._paper_repo = MariaDBPaperRepository(self.db_pool)
            else:
                from infrastructure.repositories.paper_repository import InMemoryPaperRepository
                self._paper_repo = InMemoryPaperRepository()
        return self._paper_repo

    def get_patent_repository(self) -> PatentRepository:
        if self._patent_repo is None:
            if self.is_mariadb:
                from infrastructure.repositories.patent_repository import MariaDBPatentRepository
                self._patent_repo = MariaDBPatentRepository(self.db_pool)
            else:
                from infrastructure.repositories.patent_repository import InMemoryPatentRepository
                self._patent_repo = InMemoryPatentRepository()
        return self._patent_repo

    def get_project_repository(self) -> ProjectRepository:
        if self._project_repo is None:
            if self.is_mariadb:
                from infrastructure.repositories.project_repository import MariaDBProjectRepository
                self._project_repo = MariaDBProjectRepository(self.db_pool)
            else:
                from infrastructure.repositories.project_repository import InMemoryProjectRepository
                self._project_repo = InMemoryProjectRepository()
        return self._project_repo

    def get_researcher_repository(self) -> ResearcherRepository:
        if self._researcher_repo is None:
            if self.is_mariadb:
                from infrastructure.repositories.researcher_repository import MariaDBResearcherRepository
                self._researcher_repo = MariaDBResearcherRepository(self.db_pool)
            else:
                from infrastructure.repositories.researcher_repository import InMemoryResearcherRepository
                self._researcher_repo = InMemoryResearcherRepository()
        return self._researcher_repo

    def get_technology_repository(self) -> TechnologyRepository:
        if self._technology_repo is None:
            if self.is_mariadb:
                from infrastructure.repositories.technology_repository import MariaDBTechnologyRepository
                self._technology_repo = MariaDBTechnologyRepository(self.db_pool)
            else:
                from infrastructure.repositories.technology_repository import InMemoryTechnologyRepository
                self._technology_repo = InMemoryTechnologyRepository()
        return self._technology_repo

    # ── Milvus ────────────────────────────────────────────────────────────────

    def get_vector_search_fn(self) -> Optional[Callable]:
        if self._vector_search_fn is not None:
            return self._vector_search_fn
        settings = get_settings()
        if not settings.milvus_host:
            return None
        try:
            from pymilvus import MilvusClient
            from sentence_transformers import SentenceTransformer
            from infrastructure.milvus import make_vector_search_fn, ensure_collection
            client   = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
            embedder = SentenceTransformer(settings.sentence_transformer_model)
            ensure_collection(client, settings.milvus_collection)
            self._vector_search_fn = make_vector_search_fn(
                client, embedder.encode, settings.milvus_collection
            )
        except Exception as e:
            logger.warning("[Milvus] 초기화 실패: %s", e)
            return None
        return self._vector_search_fn

    # ── Neo4j ─────────────────────────────────────────────────────────────────

    def _get_neo4j_driver(self) -> Any:
        if self._neo4j_driver is not None:
            return self._neo4j_driver
        settings = get_settings()
        if not settings.neo4j_uri:
            return None
        try:
            from neo4j import GraphDatabase
            self._neo4j_driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
            )
        except Exception as e:
            logger.warning("[Neo4j] 드라이버 초기화 실패: %s", e)
        return self._neo4j_driver

    def get_graph_query_fn(self) -> Optional[Callable]:
        driver = self._get_neo4j_driver()
        if driver is None:
            return None
        if self._graph_query_fn is None:
            from infrastructure.neo4j import make_graph_query_fn
            self._graph_query_fn = make_graph_query_fn(driver)
        return self._graph_query_fn

    def get_researcher_network_fn(self) -> Optional[Callable]:
        driver = self._get_neo4j_driver()
        if driver is None:
            return None
        if self._researcher_network_fn is None:
            from infrastructure.neo4j import make_fetch_researcher_network_fn
            self._researcher_network_fn = make_fetch_researcher_network_fn(driver)
        return self._researcher_network_fn

    def get_citation_graph_fn(self) -> Optional[Callable]:
        driver = self._get_neo4j_driver()
        if driver is None:
            return None
        if self._citation_graph_fn is None:
            from infrastructure.neo4j import make_fetch_citation_graph_fn
            self._citation_graph_fn = make_fetch_citation_graph_fn(driver)
        return self._citation_graph_fn


repository_factory = RepositoryFactory()
