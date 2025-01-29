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
}
