import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

class SfMGraphSelector:
    def __init__(self, matches_csv):
        """
        Initialize SfM graph selector with matches data
        
        Args:
            matches_csv (str or pd.DataFrame): Path to CSV or DataFrame with image matches
        """
        # Load data if a string (file path) is provided
        if isinstance(matches_csv, str):
            self.matches_df = pd.read_csv(matches_csv)
        else:
            self.matches_df = matches_csv
        
        # Initialize graph
        self.graph = self._build_image_graph()
    
    def _build_image_graph(self):
        """
        Build a graph representing image connections based on matches
        
        Returns:
            networkx.Graph: Graph of image connections
        """
        # Create an empty graph
        G = nx.Graph()
        
        # Add edges based on matched image pairs
        for _, row in self.matches_df.iterrows():
            # Add nodes
            img1 = row['img1']
            img2 = row['img2']
            
            # Add edge with match quality attributes
            G.add_edge(img1, img2, 
                       num_matches=row['num_matches'],
                       num_inliers=row['num_inliers'],
                       inlier_ratio=row['inlier_ratio'],
                       reprojection_error=row['reprojection_error'])
        
        return G
    
    def compute_node_importance(self):
        """
        Compute importance of each node in the graph
        
        Returns:
            dict: Node importance scores
        """
        # Multiple centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        
        # Weighted centrality considering match quality
        weighted_centrality = {}
        for node in self.graph.nodes():
            # Compute weighted score
            total_inliers = sum(
                self.graph[node][neighbor]['num_inliers'] 
                for neighbor in self.graph.neighbors(node)
            )
            avg_inlier_ratio = sum(
                self.graph[node][neighbor]['inlier_ratio'] 
                for neighbor in self.graph.neighbors(node)
            ) / len(list(self.graph.neighbors(node)))
            
            weighted_centrality[node] = (
                degree_centrality[node] * 0.4 +
                betweenness_centrality[node] * 0.3 +
                total_inliers / (len(list(self.graph.neighbors(node))) + 1) * 0.3
            )
        
        return weighted_centrality
    
    def find_next_best_images(self, current_reconstruction, top_k=5):
        """
        Find the next best images to add to the reconstruction
        
        Args:
            current_reconstruction (list): List of images already in the reconstruction
            top_k (int): Number of best images to return
        
        Returns:
            list: Top K images to add to reconstruction
        """
        # Compute node importance
        node_importance = self.compute_node_importance()
        
        # Filter out already reconstructed images
        candidate_images = [
            img for img in self.graph.nodes() 
            if img not in current_reconstruction
        ]
        
        # Sort candidates by importance and connection to current reconstruction
        candidate_scores = {}
        for img in candidate_images:
            # Check connections to current reconstruction
            connected_to_current = sum(
                1 for recon_img in current_reconstruction 
                if self.graph.has_edge(img, recon_img)
            )
            
            # Compute score
            candidate_scores[img] = (
                node_importance.get(img, 0) * 0.7 +
                connected_to_current * 0.3
            )
        
        # Return top K candidates
        return sorted(
            candidate_scores, 
            key=candidate_scores.get, 
            reverse=True
        )[:top_k]
    
    def visualize_graph(self, output_path='image_graph.png', max_size=1000):
        """
        Improved graph visualization with size limits and proper colorbar handling
        
        Args:
            output_path: Path to save the visualization
            max_size: Maximum number of nodes to display
        """
        # Create figure and axis explicitly
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Limit number of nodes for visualization
        if len(self.graph) > max_size:
            logging.warning(f"Graph too large ({len(self.graph)} nodes), sampling {max_size} nodes")
            subgraph = self.graph.subgraph(np.random.choice(
                list(self.graph.nodes()), max_size, replace=False))
        else:
            subgraph = self.graph
            
        # Compute layout
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
        
        # Get edge weights for coloring
        edge_weights = [subgraph[u][v]['inlier_ratio'] for u, v in subgraph.edges()]
        
        # Create a scalar mappable for the colorbar
        norm = plt.Normalize(vmin=min(edge_weights) if edge_weights else 0, 
                            vmax=max(edge_weights) if edge_weights else 1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        
        # Draw the graph
        nx.draw_networkx(
            subgraph,
            pos,
            ax=ax,
            node_size=50,
            node_color='lightblue',
            edge_color=edge_weights,
            edge_cmap=plt.cm.viridis,
            edge_vmin=norm.vmin,
            edge_vmax=norm.vmax,
            width=2,
            with_labels=True,
            font_size=8
        )
        
        # Add colorbar
        plt.colorbar(sm, ax=ax, label='Inlier Ratio')
        
        ax.set_title(f"Image Matching Graph ({len(subgraph)} nodes)")
        ax.axis('off')
        
        # Save and close
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def main():
    # Example usage
    selector = SfMGraphSelector('/teamspace/studios/this_studio/SFM/bunny_data/matching_results.csv')
    
    # Visualize the graph
    selector.visualize_graph()

if __name__ == '__main__':
    main()