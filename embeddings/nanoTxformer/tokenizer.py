import numpy as np

class scTokenizer:
    def __init__(self, adata):
        """
        initializes the tokenizer with an AnnData object, calculates normalization factors 
        (nonzero medians), and stores them for future tokenization
        
        :param adata: single-cell data stored in AnnData format
        """
        self.adata = adata
        self.gene_count = adata.shape[1]
        self.pad_token = adata.shape[1]  # define a pad token for zero-expression genes
        self.tokens = {i: i for i in range(self.gene_count)}  # assign each gene index a token
        
        # calculate and store nonzero medians for normalization based on the original dataset
        self.nonzero_medians = self.calculate_nonzero_medians(adata)

    def calculate_nonzero_medians(self, adata):
        """
        calculates the nonzero medians for each gene in the dataset
        
        :param adata: AnnData object to compute nonzero medians from
        :return: array of nonzero medians for each gene (1 if all values are zero)
        """
        X = adata.layers['counts'].toarray()
        nonzero_medians = np.ones(X.shape[1])  # initialize to 1 to avoid division by zero

        for i in range(X.shape[1]):
            nonzero_values = X[:, i][X[:, i] > 0]  # select only nonzero values for this gene
            if len(nonzero_values) > 0:
                nonzero_medians[i] = np.median(nonzero_values)  # compute the nonzero median

        return nonzero_medians

    def normalize_with_stored_medians(self, adata):
        """
        normalizes a new AnnData object using the stored nonzero medians
        
        :param adata: AnnData object to normalize
        :return: normalized AnnData object
        """
        X = adata.layers['counts'].toarray()
        normalized_X = X / self.nonzero_medians  # use the stored nonzero medians
        normalized_adata = adata.copy()
        normalized_adata.X = normalized_X
        return normalized_adata

    def tokenize_adata(self, adata):
        """
        tokenizes the entire AnnData object, normalizing using the stored nonzero medians
        
        :param adata: the AnnData object to tokenize
        :return: tokenized gene expression for each cell in the AnnData as a list of lists of tokens
        """
        # normalize the new AnnData using the stored nonzero medians
        normalized_adata = self.normalize_with_stored_medians(adata)
        
        # tokenize each cell in the normalized AnnData
        tokenized_data = []
        for cell_idx in range(normalized_adata.shape[0]):  # iterate through each cell
            # sort genes by expression and replace zero-expression genes with pad token
            tokenized_genes = self.tokenize_cell_with_pad(normalized_adata, cell_idx)
            tokenized_data.append(tokenized_genes)
        
        return tokenized_data

    def tokenize_cell_with_pad(self, adata, cell_idx):
        """
        tokenizes a single cell's gene expression by sorting and replacing zero-expression genes 
        with pad tokens
        
        :param adata: AnnData object containing the normalized data
        :param cell_idx: index of the cell to tokenize
        :return: list of tokens with pad tokens for zero-expression genes
        """
        cell_expression = adata.X[cell_idx, :]
        sorted_gene_indices = np.argsort(-cell_expression)  # sort genes in descending order of expression
        
        # replace zero-expression genes with the pad token
        tokenized_genes = []
        for gene_idx in sorted_gene_indices:
            if cell_expression[gene_idx] == 0:
                tokenized_genes.append(self.pad_token)  # use pad token for zero-expression genes
            else:
                tokenized_genes.append(self.tokens[gene_idx])  # use regular token

        return tokenized_genes
