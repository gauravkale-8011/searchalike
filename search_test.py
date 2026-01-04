import config
from src.feature_extractor import FeatureExtractor
from src.similarity_search import SimilaritySearch

query_image = "data/images/Cat/1000.jpg"

fe = FeatureExtractor(triplet_model_path=str(config.TRIPLET_MODEL_PATH))
ss = SimilaritySearch(use_faiss=True)

ss.load_index(
    index_path=config.TRIPLET_FAISS_INDEX,
    paths_path=config.TRIPLET_IMAGE_PATHS_FILE
)

query_emb = fe.extract_features(query_image)
results = ss.search(query_emb, top_k=5)

for path, score in results:
    print(path, score)
