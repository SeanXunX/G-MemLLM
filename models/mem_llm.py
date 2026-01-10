from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .latent_mem_bank import LatentMemBank


class MemLLM(nn.Module):
    """
    An memory augmented llm.
    """

    def __init__(self, model_name: str, memory_config: Optional[dict] = None):
        super().__init__()

        # Load pretrained LLM
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)

        # Freeze the LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False

        hidden_size = self.llm.config.hidden_size

        memory_config = memory_config or {}
        self.memory_bank = LatentMemBank(last_hidden_size=hidden_size, **memory_config)

        # Holding the memory information
        self.register_buffer("current_memory", None)

        # A trainable layer to inject memory information back into llm's processing
        self.memory_injection = nn.Linear(hidden_size * 2, hidden_size)

        self.current_memory: Optional[torch.Tensor] = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        use_memory: bool = True,
    ):
        # Detach the memory from the previous step's computation graph
        if self.current_memory is not None:
            self.current_memory = self.current_memory.detach()

        # 1. Get hidden states and logits from the frozen llm
        with torch.no_grad():
            llm_outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = llm_outputs.hidden_states[-1]

        # 2. Retrieve relevant memories and inject them
        if use_memory and self.current_memory is not None:
            retrieved_memory = self.memory_bank.retrieve_memory(
                self.current_memory, hidden_states
            )
            combined = torch.cat([hidden_states, retrieved_memory], dim=-1)
            enhanced_hidden = self.memory_injection(combined)
        else:
            enhanced_hidden = hidden_states

        # 3. Update the memory bank with new experience
        importance_scores = None
        if update_memory:
            new_experience = self.memory_bank.encode_experience(hidden_states)

            # Create a mask to ignore padding tokens during memory updates
            update_mask = (attention_mask == 0) if attention_mask is not None else None

            updated_memory, importance_scores = self.memory_bank.update_memory(
                self.current_memory, new_experience, mask=update_mask
            )
            self.current_memory = updated_memory

        # 4. Generate new, memory-augmented logits for training
        augmented_logits = self.llm.lm_head(enhanced_hidden)

        return {
            "enhanced_hidden_states": enhanced_hidden,
            "llm_logits": llm_outputs.logits,  # Original logits for reference
            "augmented_logits": augmented_logits,  # New logits for training
            "memory_state": self.current_memory,
            "importance_scores": importance_scores,
        }
