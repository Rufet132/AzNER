import unittest
from src.convert_to_bio import align_tokens_and_spans

class TestBioConversion(unittest.TestCase):
    
    def test_single_sentence_alignment(self):
        """
        Sənin Tapşırığın:
        1. Test üçün əllə bir test cümləsi təyin et.
        2. Həmin cümləyə uyğun əllə simulyasiya edilmiş bir 'spans' siyahısı yaz.
        3. align_tokens_and_spans funksiyasını çağır.
        4. Çıxan nəticəni gözlənilən (expected) B-PER, I-PER, B-LOC etiketləri ilə assert edərək müqayisə et.
        """
        text = "İlham Əliyev Bakıda çıxış etdi."
        
        spans = [{'start': 0, 'end': 12, 'label': 'PER'}, {'start': 13, 'end': 18, 'label': 'LOC'}]
        
        tokens, labels = align_tokens_and_spans(text, spans)
        
        expected_tokens = ["İlham", "Əliyev", "Bakıda", "çıxış", "etdi", "."]
        expected_labels = ["B-PER", "I-PER", "B-LOC", "O", "O", "O"]
        
        self.assertEqual(tokens, expected_tokens)
        self.assertEqual(labels, expected_labels)

if __name__ == "__main__":
    unittest.main()
