package fr.efrei.rag.service;

import fr.efrei.rag.domain.Document;
import fr.efrei.rag.repository.DocumentRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class DocumentService {
    private static final Logger log = LoggerFactory.getLogger(DocumentService.class);
    private final DocumentRepository documentRepository;

    public DocumentService(DocumentRepository documentRepository) {
        this.documentRepository = documentRepository;
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

    public void deleteById(Long id) {
        log.debug("Request to delete Document by ID: {}", id);
        documentRepository.deleteById(id);
    }
}