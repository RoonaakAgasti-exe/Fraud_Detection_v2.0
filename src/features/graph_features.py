"""
Graph-Based Feature Engineering for Fraud Detection

Creates network-based features using transaction graphs, centrality measures,
and community detection to identify fraudulent patterns.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple
from loguru import logger
from collections import defaultdict


class GraphFeatureEngineer:
    """Create graph-based features from transaction data"""
    
    def __init__(self, 
                 user_col: str = 'user_id',
                 merchant_col: str = 'merchant_id',
                 amount_col: str = 'amount',
                 time_col: str = 'timestamp'):
        
        self.user_col = user_col
        self.merchant_col = merchant_col
        self.amount_col = amount_col
        self.time_col = time_col
        self.graph = None
        
    def build_transaction_graph(self, df: pd.DataFrame) -> nx.Graph:
        """Build undirected graph from transactions"""
        
        logger.info("Building transaction graph...")
        
        G = nx.Graph()
        
        # Add edges between users and merchants
        for _, row in df.iterrows():
            user = f"U_{row[self.user_col]}"
            merchant = f"M_{row[self.merchant_col]}"
            
            G.add_edge(
                user, 
                merchant,
                amount=row[self.amount_col],
                timestamp=row.get(self.time_col, None)
            )
        
        self.graph = G
        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def build_directed_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """Build directed graph for money flow analysis"""
        
        logger.info("Building directed transaction graph...")
        
        G = nx.DiGraph()
        
        for _, row in df.iterrows():
            source = f"U_{row[self.user_col]}"
            target = f"M_{row[self.merchant_col]}"
            
            G.add_edge(
                source,
                target,
                amount=row[self.amount_col],
                timestamp=row.get(self.time_col, None)
            )
        
        self.graph = G
        logger.info(f"Directed graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def compute_centrality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute various centrality measures for nodes"""
        
        if self.graph is None:
            self.build_transaction_graph(df)
        
        logger.info("Computing centrality features...")
        
        # Get unique users and merchants
        users = df[self.user_col].unique()
        merchants = df[self.merchant_col].unique()
        
        # Degree centrality
        degree_cent = nx.degree_centrality(self.graph)
        
        # Betweenness centrality (identifies bridges/fraud rings)
        betweenness_cent = nx.betweenness_centrality(self.graph)
        
        # PageRank (importance based on connections)
        pagerank = nx.pagerank(self.graph, max_iter=100)
        
        # Create feature DataFrames
        user_features = pd.DataFrame({
            self.user_col: users,
            'user_degree_centrality': [degree_cent.get(f"U_{u}", 0) for u in users],
            'user_betweenness_centrality': [betweenness_cent.get(f"U_{u}", 0) for u in users],
            'user_pagerank': [pagerank.get(f"U_{u}", 0) for u in users],
        })
        
        merchant_features = pd.DataFrame({
            self.merchant_col: merchants,
            'merchant_degree_centrality': [degree_cent.get(f"M_{m}", 0) for m in merchants],
            'merchant_betweenness_centrality': [betweenness_cent.get(f"M_{m}", 0) for m in merchants],
            'merchant_pagerank': [pagerank.get(f"M_{m}", 0) for m in merchants],
        })
        
        # Merge back to original dataframe
        df = df.merge(user_features, on=self.user_col, how='left')
        df = df.merge(merchant_features, on=self.merchant_col, how='left')
        
        logger.info("Centrality features computed")
        
        return df
    
    def detect_communities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect communities (potential fraud rings) using Louvain method"""
        
        if self.graph is None:
            self.build_transaction_graph(df)
        
        logger.info("Detecting communities...")
        
        try:
            import community as community_louvain
            
            # Detect communities
            partition = community_louvain.best_partition(self.graph)
            
            # Add community labels
            users = df[self.user_col].unique()
            merchants = df[self.merchant_col].unique()
            
            user_communities = pd.DataFrame({
                self.user_col: users,
                'user_community': [partition.get(f"U_{u}", -1) for u in users]
            })
            
            merchant_communities = pd.DataFrame({
                self.merchant_col: merchants,
                'merchant_community': [partition.get(f"M_{m}", -1) for m in merchants]
            })
            
            df = df.merge(user_communities, on=self.user_col, how='left')
            df = df.merge(merchant_communities, on=self.merchant_col, how='left')
            
            # Count community sizes
            community_sizes = defaultdict(int)
            for node, comm in partition.items():
                community_sizes[comm] += 1
            
            # Add community size features
            df['user_community_size'] = df['user_community'].map(community_sizes)
            df['merchant_community_size'] = df['merchant_community'].map(community_sizes)
            
            logger.info(f"Detected {len(set(partition.values()))} communities")
            
        except ImportError:
            logger.warning("python-louvain not installed. Skipping community detection.")
            df['user_community'] = -1
            df['merchant_community'] = -1
        
        return df
    
    def compute_clustering_coefficients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute clustering coefficients to detect tightly-knit groups"""
        
        if self.graph is None:
            self.build_transaction_graph(df)
        
        logger.info("Computing clustering coefficients...")
        
        clustering = nx.clustering(self.graph)
        
        users = df[self.user_col].unique()
        merchants = df[self.merchant_col].unique()
        
        user_features = pd.DataFrame({
            self.user_col: users,
            'user_clustering_coefficient': [clustering.get(f"U_{u}", 0) for u in users]
        })
        
        merchant_features = pd.DataFrame({
            self.merchant_col: merchants,
            'merchant_clustering_coefficient': [clustering.get(f"M_{m}", 0) for m in merchants]
        })
        
        df = df.merge(user_features, on=self.user_col, how='left')
        df = df.merge(merchant_features, on=self.merchant_col, how='left')
        
        logger.info("Clustering coefficients computed")
        
        return df
    
    def compute_triangles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count triangles each node participates in"""
        
        if self.graph is None:
            self.build_transaction_graph(df)
        
        logger.info("Computing triangle counts...")
        
        triangles = nx.triangles(self.graph)
        
        users = df[self.user_col].unique()
        merchants = df[self.merchant_col].unique()
        
        user_features = pd.DataFrame({
            self.user_col: users,
            'user_triangle_count': [triangles.get(f"U_{u}", 0) for u in users]
        })
        
        merchant_features = pd.DataFrame({
            self.merchant_col: merchants,
            'merchant_triangle_count': [triangles.get(f"M_{m}", 0) for m in merchants]
        })
        
        df = df.merge(user_features, on=self.user_col, how='left')
        df = df.merge(merchant_features, on=self.merchant_col, how='left')
        
        logger.info("Triangle counts computed")
        
        return df
    
    def extract_all_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all graph-based features in one pipeline"""
        
        logger.info("Extracting all graph features...")
        
        # Build graph
        self.build_transaction_graph(df)
        
        # Extract features
        df = self.compute_centrality_features(df)
        df = self.detect_communities(df)
        df = self.compute_clustering_coefficients(df)
        df = self.compute_triangles(df)
        
        # Additional derived features
        df['centrality_ratio'] = df['user_degree_centrality'] / (df['user_clustering_coefficient'] + 1e-6)
        df['community_concentration'] = df.groupby('user_community')['user_id'].transform('count')
        
        logger.success("All graph features extracted successfully")
        
        return df


def create_graph_features(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    merchant_col: str = 'merchant_id',
    amount_col: str = 'amount',
    time_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Convenience function to create graph features
    
    Args:
        df: Transaction DataFrame
        user_col: User identifier column
        merchant_col: Merchant identifier column
        amount_col: Transaction amount column
        time_col: Timestamp column
        
    Returns:
        DataFrame with additional graph features
    """
    
    engineer = GraphFeatureEngineer(
        user_col=user_col,
        merchant_col=merchant_col,
        amount_col=amount_col,
        time_col=time_col
    )
    
    return engineer.extract_all_graph_features(df)
