package fr.efrei.rag.service;

import fr.efrei.rag.domain.Document;
import fr.efrei.rag.repository.DocumentRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
public class DocumentService {
    private static final Logger log = LoggerFactory.getLogger(DocumentService.class);
    private final DocumentRepository documentRepository;

    public DocumentService(DocumentRepository documentRepository){
        this.documentRepository = documentRepository;
    }

    public Document buildAndSave(Document document){
        log.debug("Request to buildAndSave Document : {}", document);
        return documentRepository.save(document);
    }
}
