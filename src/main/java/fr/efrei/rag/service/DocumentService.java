package fr.efrei.rag.service;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import fr.efrei.rag.domain.Document;
import fr.efrei.rag.repository.DocumentRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Service
public class DocumentService {
    private static final Logger log = LoggerFactory.getLogger(DocumentService.class);

    private static final String SYSTEM_MESSAGE_PROMPT = """
    Assistant helps the Library company customers with support questions regarding terms of service, privacy policy, and questions about support requests.
    Be brief in your answers.
    Answer ONLY with the facts listed in the list of sources below.
    If there isn't enough information below, say you don't know.
    Do not generate answers that don't use the sources below.
    If asking a clarifying question to the user would help, ask the question.
    For tabular information return it as an html table.
    Do not return markdown format.
    If the question is not in English, answer in the language used in the question.
    """;

    private final DocumentRepository documentRepository;
    private final InMemoryEmbeddingStore<TextSegment> embeddingStore;
    private final EmbeddingModel embeddingModel;
    private final ChatLanguageModel chatLanguageModel;

    public DocumentService(DocumentRepository documentRepository, InMemoryEmbeddingStore<TextSegment> embeddingStore, EmbeddingModel embeddingModel, ChatLanguageModel chatLanguageModel) {
        this.documentRepository = documentRepository;
        this.embeddingStore = embeddingStore;
        this.embeddingModel = embeddingModel;
        this.chatLanguageModel = chatLanguageModel;
    }

    public Document buildAndSave(Document document) {
        log.debug("Request to buildAndSave Document: {}", document);
        return documentRepository.save(document);
    }

    public List<Document> findAll() {
        log.debug("Request to get all Documents");
        return documentRepository.findAll();
    }

    public Document findById(Long id) {
        log.debug("Request to get Document by ID: {}", id);
        return documentRepository.findById(id).orElse(null);
    }

    public Document update(Long id, Document updatedDocument) {
        log.debug("Request to update Document with ID: {}", id);

        return documentRepository.findById(id).map(existingDocument -> {

            existingDocument.setId(updatedDocument.getId());
            existingDocument.setTitle(updatedDocument.getTitle());

            Document savedDocument = documentRepository.save(existingDocument);
            log.debug("Updated Document: {}", savedDocument);
            return savedDocument;
        }).orElseThrow(() -> new IllegalArgumentException("Document with ID " + id + " not found"));
    }


    public void deleteById(Long id) {
        log.debug("Request to delete Document by ID: {}", id);
        documentRepository.deleteById(id);
    }

    @SuppressWarnings("removal")
    public String chat(String request) {
        Embedding embeddedQuestion = embeddingModel.embed(request).content();
        List<EmbeddingMatch<TextSegment>> relevant = embeddingStore.findRelevant(embeddedQuestion, 3);
        List<ChatMessage> chatMessages = new ArrayList<>();
        chatMessages.add(SystemMessage.from(SYSTEM_MESSAGE_PROMPT));
        String userMessage = request + "\n\nSources:\n";
        for (EmbeddingMatch<TextSegment> textSegmentEmbeddingMatch : relevant) {
            userMessage += textSegmentEmbeddingMatch.embedded().text() + "\n";
        }
        chatMessages.add(UserMessage.from(userMessage));

        // Invoke the LLM
        log.info("### Invoke the LLM");
        Response<AiMessage> response = chatLanguageModel.generate(chatMessages);
        return response.content().text();
    }
}
