package fr.efrei.rag.web.rest;

import fr.efrei.rag.domain.Document;
import fr.efrei.rag.service.DocumentService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.List;

@RestController
public class DocumentResource {

    private static final Logger log = LoggerFactory.getLogger(DocumentResource.class);
    private final DocumentService documentService;

    public DocumentResource(DocumentService documentService) {
        this.documentService = documentService;
    }

    @PostMapping("/documents")
    public ResponseEntity<Document> createDocument(@RequestBody Document document) throws URISyntaxException {
        log.debug("REST request to save Document : {}", document);
        Document result = documentService.buildAndSave(document);
        return ResponseEntity.created(new URI("/documents" + result.getId())).body(result);
    }

    @GetMapping("/documents")
    public List<Document> getDocuments() {
        return documentService.findAll();
    }

    @GetMapping("/document/{value}")
    public Document getDocumentById(@PathVariable(value = "value", required = false)final String value) {
        Long id = value == null ? null : Long.valueOf(value);
        return documentService.findById(id);
    }

    @PutMapping("/document/{value}")
    public ResponseEntity<Document> updateDocument(@PathVariable(value = "value") final String value,@RequestBody Document updatedDocument) {
        log.debug("REST request to update Document : {}", value);

        Long id = value == null ? null : Long.valueOf(value);

        try {
            Document result = documentService.update(id, updatedDocument);
            return ResponseEntity.ok(result);
        } catch (IllegalArgumentException e) {
            log.error("Document not found: {}", e.getMessage());
            return ResponseEntity.notFound().build();
        }
    }

    @DeleteMapping("/document/{value}")
    public ResponseEntity<Void> deleteDocumentById(@PathVariable(value = "value", required = false)final String value) {
        Long id = value == null ? null : Long.valueOf(value);
        documentService.deleteById(id);
        return ResponseEntity.noContent().build();
    }
}
