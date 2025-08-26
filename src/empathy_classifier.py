import torch
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from models.models import BiEncoderAttentionWithRationaleClassification

class EmpathyClassifier():
    def __init__(self, 
                 device,
                 ER_model_path='output/sample.pth',
                 IP_model_path='output/sample.pth',
                 EX_model_path='output/sample.pth',
                 batch_size=1):
        # modern tokenizer (no do_lower_case for RoBERTa)
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)
        self.batch_size = batch_size
        self.device = device

        self.model_ER = BiEncoderAttentionWithRationaleClassification()
        self.model_IP = BiEncoderAttentionWithRationaleClassification()
        self.model_EX = BiEncoderAttentionWithRationaleClassification()

        ER_weights = torch.load(ER_model_path, map_location=device)
        self.model_ER.load_state_dict(ER_weights)

        IP_weights = torch.load(IP_model_path, map_location=device)
        self.model_IP.load_state_dict(IP_weights)

        EX_weights = torch.load(EX_model_path, map_location=device)
        self.model_EX.load_state_dict(EX_weights)

        self.model_ER.to(self.device)
        self.model_IP.to(self.device)
        self.model_EX.to(self.device)

    def _encode(self, texts, max_len=64):
        enc = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    def predict_empathy(self, seeker_posts, response_posts):
        input_ids_SP, attention_masks_SP = self._encode(seeker_posts)
        input_ids_RP, attention_masks_RP = self._encode(response_posts)

        dataset = TensorDataset(input_ids_SP, attention_masks_SP, input_ids_RP, attention_masks_RP)
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size)

        self.model_ER.eval(); self.model_IP.eval(); self.model_EX.eval()

        for batch in dataloader:
            b_input_ids_SP = batch[0].to(self.device)
            b_input_mask_SP = batch[1].to(self.device)
            b_input_ids_RP = batch[2].to(self.device)
            b_input_mask_RP = batch[3].to(self.device)

            with torch.no_grad():
                (logits_empathy_ER, logits_rationale_ER,) = self.model_ER(
                    input_ids_SP=b_input_ids_SP, input_ids_RP=b_input_ids_RP,
                    token_type_ids_SP=None, token_type_ids_RP=None,
                    attention_mask_SP=b_input_mask_SP, attention_mask_RP=b_input_mask_RP
                )
                (logits_empathy_IP, logits_rationale_IP,) = self.model_IP(
                    input_ids_SP=b_input_ids_SP, input_ids_RP=b_input_ids_RP,
                    token_type_ids_SP=None, token_type_ids_RP=None,
                    attention_mask_SP=b_input_mask_SP, attention_mask_RP=b_input_mask_RP
                )
                (logits_empathy_EX, logits_rationale_EX,) = self.model_EX(
                    input_ids_SP=b_input_ids_SP, input_ids_RP=b_input_ids_RP,
                    token_type_ids_SP=None, token_type_ids_RP=None,
                    attention_mask_SP=b_input_mask_SP, attention_mask_RP=b_input_mask_RP
                )

            logits_empathy_ER = logits_empathy_ER.detach().cpu().numpy().tolist()
            predictions_ER = np.argmax(logits_empathy_ER, axis=1).flatten()

            logits_empathy_IP = logits_empathy_IP.detach().cpu().numpy().tolist()
            predictions_IP = np.argmax(logits_empathy_IP, axis=1).flatten()

            logits_empathy_EX = logits_empathy_EX.detach().cpu().numpy().tolist()
            predictions_EX = np.argmax(logits_empathy_EX, axis=1).flatten()

            logits_rationale_ER = logits_rationale_ER.detach().cpu().numpy()
            predictions_rationale_ER = np.argmax(logits_rationale_ER, axis=2)

            logits_rationale_IP = logits_rationale_IP.detach().cpu().numpy()
            predictions_rationale_IP = np.argmax(logits_rationale_IP, axis=2)

            logits_rationale_EX = logits_rationale_EX.detach().cpu().numpy()
            predictions_rationale_EX = np.argmax(logits_rationale_EX, axis=2)

        return (logits_empathy_ER, predictions_ER,
                logits_empathy_IP, predictions_IP,
                logits_empathy_EX, predictions_EX,
                logits_rationale_ER, predictions_rationale_ER,
                logits_rationale_IP, predictions_rationale_IP,
                logits_rationale_EX, predictions_rationale_EX)
