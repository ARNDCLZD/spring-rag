package fr.efrei.rag.web.rest;

import fr.efrei.rag.domain.Document;
import fr.efrei.rag.service.DocumentService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.net.URI;
import java.net.URISyntaxException;

@RestController
public class DocumentResource {

    private final DocumentService documentService;

    public DocumentResource(DocumentService documentService) {
        this.documentService = documentService;
    }

    @PostMapping("/documents")
    public ResponseEntity<Document> createDocument(@RequestBody Document document) throws URISyntaxException {
        Document result = documentService.buildAndSave(document);
        return ResponseEntity
                .created(new URI("/documents" + result.getId()))
                .body(result);
    }
}
