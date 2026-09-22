import unittest
import os
from src.ingestion.loader import load_single_docx
from src.retrieval.retriever import expand_query,get_related_subjects,is_context_relevant,is_garbled_ocr
from src.utils.helpers import load_app_config,is_short_query,SUBJECT_METADATA,SUBJECT_SCOPE_CONTEXT
from src.llm.llm_client import is_ollama_online


class TestOmniDocRAG(unittest.TestCase):

    def test_config_loading(self):
        cfg = load_app_config()
        self.assertIn("app",cfg)
        self.assertIn("database",cfg)
        self.assertIn("retrieval",cfg)

    def test_stm_abbreviation_expansion(self):
        # In STM, CN must map to Control Flow Graph
        expanded=expand_query("explain CN",active_subject="STM Notes")
        self.assertIn("Control Flow Graph",expanded)

        # In STM, JF must map to Junction Flow
        expanded_jf=expand_query("what is JF",active_subject="STM Notes")
        self.assertIn("Junction",expanded_jf)

    def test_ml_abbreviation_expansion(self):
        # In ML, DA must map to Data Analysis
        expanded=expand_query("explain DA",active_subject="ML notes")
        self.assertIn("Data Analysis",expanded)

        # In ML, SVM must map to Support Vector Machine
        expanded_svm=expand_query("what is SVM",active_subject="ML notes")
        self.assertIn("Support Vector Machine",expanded_svm)

    def test_daa_abbreviation_expansion(self):
        # In DAA, DA must map to Design and Analysis of Algorithms
        expanded=expand_query("what is DA",active_subject="DAA Notes")
        self.assertIn("Design and Analysis of Algorithms",expanded)

    def test_acs_lab_expansion(self):
        expanded=expand_query("what is ACS Lab",active_subject="ACS Lab")
        self.assertIn("Advanced Communication Systems",expanded)

    def test_msf_and_cns_lab_query_expansion(self):
        msf_query=expand_query("What is Management Science?",active_subject="MSF")
        self.assertIn("Management Science and Finance",msf_query)

        cns_lab_query=expand_query("List all CNS lab experiments",active_subject="CNS Lab")
        self.assertIn("Cryptography and Network Security",cns_lab_query)

    def test_scanned_subject_scope_fallback_is_useful(self):
        self.assertIn("Management science applies",SUBJECT_SCOPE_CONTEXT["MSF"])
        self.assertIn("Economics studies",SUBJECT_SCOPE_CONTEXT["POE"])

    def test_case_insensitive_subject_resolution(self):
        expanded=expand_query("explain DA",active_subject="ml notes")
        self.assertIn("Data Analysis",expanded)

        expanded_acs=expand_query("what is ACS Lab",active_subject="acs lab")
        self.assertIn("Advanced Communication Systems",expanded_acs)

        long_acs_context=(
            "Experiment 1: Time Division Multiplexing in optical communication systems. "
            "Experiment 2: Optical Fiber Link Setup and attenuation measurement for signal loss."
        )
        is_rel,msg=is_context_relevant("What is ACS Lab",long_acs_context,"acs lab")
        self.assertTrue(is_rel)

    def test_economics_definition_is_not_rejected_for_related_terms(self):
        short_context="Economic principles explain how households and firms allocate scarce resources."
        is_rel,_=is_context_relevant("What is Economics?",short_context,"POE")
        self.assertTrue(is_rel)

        richer_context=(
            "Economic principles explain how households and firms allocate scarce resources. "
            "The study of demand, supply, and market equilibrium forms the core of economic analysis."
        )
        is_rel,_=is_context_relevant("What is Economics?",richer_context,"POE")
        self.assertTrue(is_rel)

    def test_no_cross_contamination(self):
        # Asking CN in Java should not expand to Computer Networks
        expanded = expand_query("what is CN",active_subject="Java")
        self.assertNotIn("Computer Networks",expanded)

    def test_cn_osi_concept_expansion(self):
        osi_query = expand_query("what are the 7 layers", active_subject="CN Notes")
        self.assertIn("Open Systems Interconnection", osi_query)
        self.assertIn("Application", osi_query)
        self.assertIn("Physical", osi_query)

        tcp_query = expand_query("explain TCP/IP model", active_subject="CN Notes")
        self.assertIn("TCP/IP model", tcp_query)
        self.assertIn("transport", tcp_query)

    def test_short_query_detection(self):
        self.assertTrue(is_short_query("CN"))
        self.assertTrue(is_short_query("what is DA"))
        self.assertFalse(is_short_query("explain the differences between linear and logistic regression in machine learning"))

    def test_ocr_garbage_detection(self):
        self.assertTrue(is_garbled_ocr("$$#@!!%^& 12 3"))
        self.assertFalse(is_garbled_ocr("A Control Flow Graph (CFG) is a graphical representation of control flow."))

    def test_subject_mismatch_relevance_gate(self):
        # Asking Data Analysis in Java with unrelated context
        fake_java_context = "Java is an object oriented language. Classes and objects encapsulate data and functions."
        is_rel, msg = is_context_relevant("Explain about Data Analysis", fake_java_context, "Java")
        self.assertFalse(is_rel)
        self.assertIn("Java Programming", msg)

    def test_subject_meta_query_relevance(self):
        # Asking about the subject itself should be relevant
        fake_acs_context = "Experiment 1: Time Division Multiplexing. Experiment 2: Optical Fiber Link Setup."
        is_rel, msg = is_context_relevant("What is ACS Lab", fake_acs_context, "ACS Lab")
        self.assertTrue(is_rel)

    def test_ollama_socket_check_never_crashes(self):
        # Must return boolean and never raise an unhandled exception
        online = is_ollama_online(timeout=0.2)
        self.assertIsInstance(online, bool)

    def test_corrupt_docx_media_does_not_discard_text(self):
        paths = [
            "PDF_Data/Data structure/DS UNIT Ill.docx",
            "PDF_Data/Data structure/DS UNIT IV(2).docx",
        ]
        for path in paths:
            if os.path.exists(path):
                documents = load_single_docx(path)
                self.assertTrue(documents)
                self.assertGreater(len(documents[0].page_content), 100)


    def test_all_groq_models_defined(self):
        from src.llm.llm_client import ALL_GROQ_MODELS
        self.assertEqual(len(ALL_GROQ_MODELS), 9)
        self.assertIn("openai/gpt-oss-120b", ALL_GROQ_MODELS)
        self.assertIn("llama-3.3-70b-versatile", ALL_GROQ_MODELS)
        self.assertIn("meta-llama/llama-4-scout-17b-16e-instruct", ALL_GROQ_MODELS)
        self.assertIn("qwen/qwen3.6-27b", ALL_GROQ_MODELS)
        self.assertIn("openai/gpt-oss-20b", ALL_GROQ_MODELS)
        self.assertIn("llama-3.1-8b-instant", ALL_GROQ_MODELS)
        self.assertIn("groq/compound-mini", ALL_GROQ_MODELS)
        self.assertIn("groq/compound", ALL_GROQ_MODELS)
        self.assertIn("allam-2-7b", ALL_GROQ_MODELS)


if __name__=="__main__":
    unittest.main()