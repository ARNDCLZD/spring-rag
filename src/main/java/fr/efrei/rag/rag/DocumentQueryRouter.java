package fr.efrei.rag.rag;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

import static java.util.Collections.emptyList;
import static java.util.Collections.singletonList;

public class DocumentQueryRouter implements QueryRouter {
    private static final Logger logger = LoggerFactory.getLogger(DocumentQueryRouter.class);

    private static final PromptTemplate PROMPT_TEMPLATE = PromptTemplate.from("""
			Is the following query related to one or more documents?
			Answer only 'yes' or 'no'.
			Query: {{it}}
			""");

    private final ContentRetriever documentContentRetriever;

    private final ChatLanguageModel chatLanguageModel;

    public DocumentQueryRouter(ChatLanguageModel chatLanguageModel, ContentRetriever documentContentRetriever) {
        this.chatLanguageModel = chatLanguageModel;
        this.documentContentRetriever = documentContentRetriever;
    }

    @Override
    public Collection<ContentRetriever> route(Query query) {
        Prompt prompt = PROMPT_TEMPLATE.apply(query.text());

        AiMessage aiMessage = chatLanguageModel.generate(prompt.toUserMessage()).content();
        logger.debug("LLM decided: {}", aiMessage.text());

        if (aiMessage.text().toLowerCase().contains("yes")) {
            return singletonList(documentContentRetriever);
        }
        return emptyList();
    }

}