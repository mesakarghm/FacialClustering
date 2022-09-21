import os
import shutil
from os.path import join, exists, basename


from PIL import Image
from facenet.feature_extractor import FacenetFeatureExtractor


def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    import numpy as np
    if len(face_encodings) == 0:
        return np.empty((0))

    # return 1/np.linalg.norm(face_encodings - face_to_compare, axis=1)
    return np.sum(face_encodings * face_to_compare, axis=1)


class ClusteringChineseWhisperers:
    def __init__(self):
        self.facenet = FacenetFeatureExtractor()

    def compute_facial_encodings(self, paths, batch_size):
        images = [Image.open(path) for path in paths]
        nrof_images = len(paths)
        print(f"Got {nrof_images} images...")
        facial_encodings = {}
        # for pth in paths:
        #     facial_encodings[pth] = self.facenet.extract_face_embedding(Image.open(pth),normalize=True)[0]
        embeddings = self.facenet.extract_face_embeddings_batch(images, normalize = True, batch_size = batch_size)
        facial_encodings = {}
        for x in range(nrof_images):
            facial_encodings[paths[x]] = embeddings[x, :]
        return facial_encodings

    def cluster_facial_encodings(self, facial_encodings):
        if len(facial_encodings) <= 1:
            print("Number of facial encodings must be greater than one, can't cluster")
            return []

        # Only use the chinese whispers algorithm for now
        sorted_clusters = self._chinese_whispers(facial_encodings.items())
        return sorted_clusters

    def _chinese_whispers(self, encoding_list, threshold=0.6, iterations=50):
        """ Chinese Whispers Algorithm

        Modified from Alex Loveless' implementation,
        http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/

        Inputs:
            encoding_list: a list of facial encodings from face_recognition
            threshold: facial match threshold,default 0.6
            iterations: since chinese whispers is an iterative algorithm, number of times to iterate

        Outputs:
            sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
                sorted by largest cluster to smallest
        """

        #from face_recognition.api import _face_distance
        from random import shuffle
        import networkx as nx
        # Create graph
        nodes = []
        edges = []

        image_paths, encodings = zip(*encoding_list)

        if len(encodings) <= 1:
            print ("No enough encodings to cluster!")
            return []

        for idx, face_encoding_to_check in enumerate(encodings):
            # Adding node of facial encoding
            node_id = idx+1

            # Initialize 'cluster' to unique value (cluster of itself)
            node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx]})
            nodes.append(node)

            # Facial encodings to compare
            if (idx+1) >= len(encodings):
                # Node is last element, don't create edge
                break

            compare_encodings = encodings[idx+1:]
            distances = face_distance(compare_encodings, face_encoding_to_check)
            encoding_edges = []
            for i, distance in enumerate(distances):
                if distance > threshold:
                    # Add edge if facial match
                    edge_id = idx+i+2
                    encoding_edges.append((node_id, edge_id, {'weight': distance}))

            edges = edges + encoding_edges

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        # Iterate
        for _ in range(0, iterations):
            cluster_nodes = G.nodes()
            shuffle(cluster_nodes)
            for node in cluster_nodes:
                neighbors = G[node]
                clusters = {}

                for ne in neighbors:
                    if isinstance(ne, int):
                        if G.node[ne]['cluster'] in clusters:
                            clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
                        else:
                            clusters[G.node[ne]['cluster']] = G[node][ne]['weight']

                # find the class with the highest edge weight sum
                edge_weight_sum = 0
                max_cluster = 0
                #use the max sum of neighbor weights class as current node's class
                for cluster in clusters:
                    if clusters[cluster] > edge_weight_sum:
                        edge_weight_sum = clusters[cluster]
                        max_cluster = cluster

                # set the class of target node to the winning local class
                G.node[node]['cluster'] = max_cluster

        clusters = {}

        # Prepare cluster output
        for (_, data) in G.node.items():
            cluster = data['cluster']
            path = data['path']

            if cluster:
                if cluster not in clusters:
                    clusters[cluster] = []
                clusters[cluster].append(path)

        # Sort cluster output
        sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

        return sorted_clusters

    def run(self, input_dir, output_dir, batch_size):
        image_list = os.listdir(input_dir)
        images_list = [os.path.join(input_dir, img) for img in image_list]
        embeddings = self.compute_facial_encodings(images_list, batch_size)
        sorted_clusters = self.cluster_facial_encodings(embeddings)
        num_clusters = len(sorted_clusters)
        for idx, cluster in enumerate(sorted_clusters):
            # save all the cluster
            cluster_dir = join(output_dir, str(idx))
            if not exists(cluster_dir):
                os.makedirs(cluster_dir)
            for path in cluster:
                shutil.copy(path, join(cluster_dir, basename(path)))



def parse_args():
    """Parse input arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Get a shape mesh (t-pose)')
    parser.add_argument('--batch_size', type=int, help='batch size', required=True, default =30)
    parser.add_argument('--input', type=str, help='Input dir of images', required=True)
    parser.add_argument('--output', type=str, help='Output dir of clusters', required=True)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    clusterer = ClusteringChineseWhisperers()
    clusterer.run(args.input, args.output, args.batch_size)