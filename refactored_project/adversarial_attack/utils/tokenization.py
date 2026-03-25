"""Tokenization utilities"""

import torch
from typing import List, Union, Optional
from transformers import PreTrainedTokenizer


class TokenizationHelper:
    """Helper class for tokenization operations"""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize tokenization helper

        Args:
            tokenizer: HuggingFace tokenizer
        """
        self.tokenizer = tokenizer

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize(
        self,
        text: Union[str, List[str]],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> dict:
        """
        Tokenize text

        Args:
            text: Input text or list of texts
            return_tensors: Return tensor type ('pt', 'tf', 'np')
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            add_special_tokens: Whether to add special tokens

        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        return self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )

    def decode(
        self,
        token_ids: Union[torch.Tensor, List[int]],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text

        Args:
            token_ids: Token IDs tensor or list
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() == 1:
                return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
            else:
                return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_token_embeddings(
        self,
        model,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Get embeddings for token IDs

        Args:
            model: Model with embedding layer
            token_ids: Token IDs tensor

        Returns:
            Embedding tensor
        """
        return model.get_input_embeddings()(token_ids)

    def get_approximate_tokens_from_embeddings(
        self,
        model,
        embeddings: torch.Tensor,
        method: str = "cosine"
    ) -> torch.Tensor:
        """
        Find closest tokens to given embeddings

        Args:
            model: Model with embedding layer
            embeddings: Embedding tensor (batch, seq_len, hidden_dim)
            method: Similarity method ('cosine' or 'euclidean')

        Returns:
            Token IDs tensor
        """
        embedding_layer = model.get_input_embeddings()
        embedding_weight = embedding_layer.weight  # (vocab_size, hidden_dim)

        # Normalize for cosine similarity
        if method == "cosine":
            embeddings_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
            weight_norm = embedding_weight / embedding_weight.norm(dim=1, keepdim=True)

            # Compute similarity: (batch, seq_len, hidden_dim) @ (hidden_dim, vocab_size)
            # -> (batch, seq_len, vocab_size)
            similarity = torch.matmul(
                embeddings_norm.view(-1, embeddings.size(-1)),
                weight_norm.t()
            )

        elif method == "euclidean":
            # Compute negative euclidean distance
            # (batch * seq_len, vocab_size)
            distances = torch.cdist(
                embeddings.view(-1, embeddings.size(-1)),
                embedding_weight,
                p=2
            )
            similarity = -distances

        else:
            raise ValueError(f"Unknown method: {method}")

        # Get most similar tokens
        approximate_token_ids = torch.argmax(similarity, dim=-1).view(
            embeddings.size(0), embeddings.size(1)
        )

        return approximate_token_ids

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.tokenizer)

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID"""
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID"""
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID"""
        return self.tokenizer.bos_token_id


def create_rejection_word_mask(
    tokenizer: PreTrainedTokenizer,
    rejection_words: List[str],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create a mask for rejection words

    Args:
        tokenizer: HuggingFace tokenizer
        rejection_words: List of rejection words
        device: Device to place mask on
        dtype: Data type for mask

    Returns:
        Binary mask tensor of shape (vocab_size,)
    """
    # Expand rejection words with variations
    expanded_words = set()
    for word in rejection_words:
        expanded_words.add(word)
        expanded_words.add(word.upper())
        expanded_words.add(word.lower())
        expanded_words.add(word.capitalize())

    # Tokenize all rejection words
    rejection_text = " ".join(list(expanded_words))
    rejection_token_ids = tokenizer.encode(rejection_text, add_special_tokens=False)

    # Create mask
    vocab_size = len(tokenizer)
    mask = torch.zeros(vocab_size, dtype=dtype, device=device)
    mask[rejection_token_ids] = 1.0

    return mask


# Default rejection words (from original code)
DEFAULT_REJECTION_WORDS = [
    "sorry", "apologetic", "regretful", "unethical", "immoral", "dishonest",
    "illegal", "unlawful", "illicit", "harmful", "detrimental", "injurious",
    "afraid", "fearful", "apprehensive", "cannot", "unable", "apologize",
    "regret", "incapable", "powerless", "not", "no", "nay", "responsible",
    "accountable", "liable", "ethic", "morality", "moral", "legal", "lawful",
    "legitimate", "ethical", "principled", "fulfill", "accomplish", "achieve",
    "just", "fair", "equitable", "trustworthy", "reliable", "dependable",
    "safe", "can't", "but", "against",
]
