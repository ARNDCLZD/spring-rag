package fr.efrei.rag.domain;

import jakarta.persistence.*;

@Entity
public class Document {
    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE)
    @Column(nullable = false)
    private Long id;

    private String title;

    public void setId(Long id) {
        this.id = id;
    }

    public Long getId() {
        return id;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getTitle() {
        return title;
    }
}
