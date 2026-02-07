// Diep.io Tank Simulation - Part 3: Rendering Functions
#include "diep_simulation_part2.c"

void draw_circle(SDL_Renderer *renderer, int x, int y, int radius, uint32_t color) {
    Uint8 r = (color >> 24) & 0xFF;
    Uint8 g = (color >> 16) & 0xFF;
    Uint8 b = (color >> 8) & 0xFF;
    Uint8 a = color & 0xFF;
    
    SDL_SetRenderDrawColor(renderer, r, g, b, a);
    
    for (int w = 0; w < radius * 2; w++) {
        for (int h = 0; h < radius * 2; h++) {
            int dx = radius - w;
            int dy = radius - h;
            if ((dx * dx + dy * dy) <= (radius * radius)) {
                SDL_RenderDrawPoint(renderer, x + dx, y + dy);
            }
        }
    }
}

void draw_agent(GameState *game, Agent *agent) {
    float screen_x = world_to_screen_x(game, agent->x);
    float screen_y = world_to_screen_y(game, agent->y);
    float screen_radius = agent->radius * game->camera.zoom;
    
    // Draw agent body
    draw_circle(game->renderer, (int)screen_x, (int)screen_y, (int)screen_radius, agent->color);
    
    // Draw cannon(s) based on tank type
    Uint8 r = (agent->color >> 24) & 0xFF;
    Uint8 g = (agent->color >> 16) & 0xFF;
    Uint8 b = (agent->color >> 8) & 0xFF;
    Uint8 a = agent->color & 0xFF;
    
    SDL_SetRenderDrawColor(game->renderer, r, g, b, a);
    
    float cannon_length = (agent->radius + 15) * game->camera.zoom;
    
    switch (agent->tank_type) {
        case TANK_TWIN:
            // Draw two cannons
            for (int i = -1; i <= 1; i += 2) {
                float offset = i * screen_radius * 0.5f;
                float cannon_start_x = screen_x + cosf(agent->angle + M_PI/2) * offset;
                float cannon_start_y = screen_y + sinf(agent->angle + M_PI/2) * offset;
                float cannon_end_x = cannon_start_x + cosf(agent->angle) * cannon_length;
                float cannon_end_y = cannon_start_y + sinf(agent->angle) * cannon_length;
                SDL_RenderDrawLine(game->renderer, (int)cannon_start_x, (int)cannon_start_y, 
                                 (int)cannon_end_x, (int)cannon_end_y);
            }
            break;
            
        case TANK_MACHINE:
            // Draw machine gun barrel
            {
                float cannon_end_x = screen_x + cosf(agent->angle) * cannon_length * 1.2f;
                float cannon_end_y = screen_y + sinf(agent->angle) * cannon_length * 1.2f;
                SDL_RenderDrawLine(game->renderer, (int)screen_x, (int)screen_y, 
                                 (int)cannon_end_x, (int)cannon_end_y);
            }
            break;
            
        case TANK_DESTROYER:
            // Draw large cannon
            {
                // Draw thicker cannon by drawing multiple lines
                for (int i = -2; i <= 2; i++) {
                    float offset = i * 2;
                    float start_x = screen_x + cosf(agent->angle + M_PI/2) * offset;
                    float start_y = screen_y + sinf(agent->angle + M_PI/2) * offset;
                    float end_x = start_x + cosf(agent->angle) * cannon_length * 1.5f;
                    float end_y = start_y + sinf(agent->angle) * cannon_length * 1.5f;
                    SDL_RenderDrawLine(game->renderer, (int)start_x, (int)start_y, 
                                     (int)end_x, (int)end_y);
                }
            }
            break;
            
        default: // TANK_BASIC, TANK_SNIPER
            {
                float cannon_end_x = screen_x + cosf(agent->angle) * cannon_length;
                float cannon_end_y = screen_y + sinf(agent->angle) * cannon_length;
                SDL_RenderDrawLine(game->renderer, (int)screen_x, (int)screen_y, 
                                 (int)cannon_end_x, (int)cannon_end_y);
            }
            break;
    }
    
    // Draw level indicator
    if (agent->level > 1) {
        SDL_SetRenderDrawColor(game->renderer, 255, 255, 255, 255);
        char level_text[10];
        snprintf(level_text, sizeof(level_text), "%d", agent->level);
        
        if (game->font) {
            SDL_Color textColor = {255, 255, 255, 255};
            SDL_Surface *textSurface = TTF_RenderText_Solid(game->font, level_text, textColor);
            if (textSurface) {
                SDL_Texture *textTexture = SDL_CreateTextureFromSurface(game->renderer, textSurface);
                if (textTexture) {
                    SDL_Rect textRect = {(int)(screen_x - textSurface->w/2), 
                                       (int)(screen_y - screen_radius - 20), 
                                       textSurface->w, textSurface->h};
                    SDL_RenderCopy(game->renderer, textTexture, NULL, &textRect);
                    SDL_DestroyTexture(textTexture);
                }
                SDL_FreeSurface(textSurface);
            }
        }
    }
    
    // Draw agent ID
    SDL_SetRenderDrawColor(game->renderer, 255, 255, 255, 255);
    char id_text[10];
    snprintf(id_text, sizeof(id_text), "%d", agent->agent_id);
    
    if (game->font) {
        SDL_Color textColor = {255, 255, 255, 255};
        SDL_Surface *textSurface = TTF_RenderText_Solid(game->font, id_text, textColor);
        if (textSurface) {
            SDL_Texture *textTexture = SDL_CreateTextureFromSurface(game->renderer, textSurface);
            if (textTexture) {
                SDL_Rect textRect = {(int)(screen_x - textSurface->w/2), 
                                   (int)(screen_y + screen_radius + 5), 
                                   textSurface->w, textSurface->h};
                SDL_RenderCopy(game->renderer, textTexture, NULL, &textRect);
                SDL_DestroyTexture(textTexture);
            }
            SDL_FreeSurface(textSurface);
        }
    }
}

void draw_bullet(GameState *game, Bullet *bullet) {
    float screen_x = world_to_screen_x(game, bullet->x);
    float screen_y = world_to_screen_y(game, bullet->y);
    float screen_radius = bullet->radius * game->camera.zoom;
    
    draw_circle(game->renderer, (int)screen_x, (int)screen_y, 
              (int)screen_radius, bullet->color);
}

void draw_shape(GameState *game, Shape *shape) {
    float screen_x = world_to_screen_x(game, shape->x);
    float screen_y = world_to_screen_y(game, shape->y);
    float screen_radius = shape->radius * game->camera.zoom;
    
    draw_circle(game->renderer, (int)screen_x, (int)screen_y, 
              (int)screen_radius, shape->color);
}

void draw_ui(SDL_Renderer *renderer, TTF_Font *font, GameState *game) {
    if (!font) {
        // Draw simple UI using rectangles when no font is available
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        
        // Draw simulation info
        SDL_Rect info_box = {10, 10, 300, 80};
        SDL_RenderDrawRect(renderer, &info_box);
        
        return;
    }
    
    // Draw simulation stats
    char stats_text[200];
    int alive_agents = 0;
    int total_level = 0;
    int total_score = 0;
    
    for (int i = 0; i < MAX_AGENTS; i++) {
        if (game->agents[i].alive) {
            alive_agents++;
            total_level += game->agents[i].level;
            total_score += game->agents[i].score;
        }
    }
    
    snprintf(stats_text, sizeof(stats_text), 
             "Generation: %d | Alive: %d/%d | Avg Level: %.1f | Total Score: %d", 
             game->generation, alive_agents, MAX_AGENTS, 
             alive_agents > 0 ? (float)total_level / alive_agents : 0.0f, total_score);
    
    SDL_Color textColor = {255, 255, 255, 255};
    SDL_Surface *textSurface = TTF_RenderText_Solid(font, stats_text, textColor);
    if (textSurface) {
        SDL_Texture *textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
        if (textTexture) {
            SDL_Rect textRect = {10, 10, textSurface->w, textSurface->h};
            SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
            SDL_DestroyTexture(textTexture);
        }
        SDL_FreeSurface(textSurface);
    }
    
    // Draw top performers
    char top_text[200];
    int best_agent = 0;
    int best_score = -1;
    
    for (int i = 0; i < MAX_AGENTS; i++) {
        if (game->agents[i].score > best_score) {
            best_score = game->agents[i].score;
            best_agent = i;
        }
    }
    
    if (best_score >= 0) {
        snprintf(top_text, sizeof(top_text), 
                 "Leader: Agent %d (Score: %d, Level: %d, Type: %s)", 
                 best_agent, game->agents[best_agent].score, game->agents[best_agent].level,
                 game->agents[best_agent].tank_type == TANK_BASIC ? "Basic" :
                 game->agents[best_agent].tank_type == TANK_TWIN ? "Twin" :
                 game->agents[best_agent].tank_type == TANK_SNIPER ? "Sniper" :
                 game->agents[best_agent].tank_type == TANK_MACHINE ? "Machine" : "Destroyer");
        
        textSurface = TTF_RenderText_Solid(font, top_text, textColor);
        if (textSurface) {
            SDL_Texture *textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
            if (textTexture) {
                SDL_Rect textRect = {10, 40, textSurface->w, textSurface->h};
                SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
                SDL_DestroyTexture(textTexture);
            }
            SDL_FreeSurface(textSurface);
        }
    }
    
    // Draw controls
    char controls_text[] = "Controls: P=Pause, +/-=Zoom, Space=Focus Random Agent, ESC=Quit";
    textSurface = TTF_RenderText_Solid(font, controls_text, textColor);
    if (textSurface) {
        SDL_Texture *textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
        if (textTexture) {
            SDL_Rect textRect = {10, SCREEN_HEIGHT - 25, textSurface->w, textSurface->h};
            SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
            SDL_DestroyTexture(textTexture);
        }
        SDL_FreeSurface(textSurface);
    }
    
    // Draw pause indicator
    if (game->paused) {
        char pause_text[] = "PAUSED - Grid Visualization Enabled";
        SDL_Color pauseColor = {255, 255, 0, 255};
        SDL_Surface *pauseSurface = TTF_RenderText_Solid(font, pause_text, pauseColor);
        if (pauseSurface) {
            SDL_Texture *pauseTexture = SDL_CreateTextureFromSurface(renderer, pauseSurface);
            if (pauseTexture) {
                SDL_Rect pauseRect = {SCREEN_WIDTH / 2 - 150, SCREEN_HEIGHT / 2 - 15, 
                                   pauseSurface->w, pauseSurface->h};
                SDL_RenderCopy(renderer, pauseTexture, NULL, &pauseRect);
                SDL_DestroyTexture(pauseTexture);
            }
            SDL_FreeSurface(pauseSurface);
        }
    }
    
    // Draw fitness rankings (top 5)
    SDL_Color fitnessColor = {0, 255, 255, 255};
    char fitness_title[] = "Top 5 Agents by Fitness:";
    textSurface = TTF_RenderText_Solid(font, fitness_title, fitnessColor);
    if (textSurface) {
        SDL_Texture *textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
        if (textTexture) {
            SDL_Rect textRect = {SCREEN_WIDTH - 250, 10, textSurface->w, textSurface->h};
            SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
            SDL_DestroyTexture(textTexture);
        }
        SDL_FreeSurface(textSurface);
    }
    
    // Sort agents by fitness for display
    int sorted_indices[MAX_AGENTS];
    for (int i = 0; i < MAX_AGENTS; i++) {
        sorted_indices[i] = i;
    }
    
    for (int i = 0; i < MAX_AGENTS - 1; i++) {
        for (int j = i + 1; j < MAX_AGENTS; j++) {
            if (game->fitness_scores[sorted_indices[j]] > game->fitness_scores[sorted_indices[i]]) {
                int temp = sorted_indices[i];
                sorted_indices[i] = sorted_indices[j];
                sorted_indices[j] = temp;
            }
        }
    }
    
    for (int i = 0; i < 5 && i < MAX_AGENTS; i++) {
        int agent_idx = sorted_indices[i];
        char fitness_text[100];
        snprintf(fitness_text, sizeof(fitness_text), 
                 "%d. Agent %d: %.1f (Lvl %d)", 
                 i + 1, agent_idx, game->fitness_scores[agent_idx], game->agents[agent_idx].level);
        
        textSurface = TTF_RenderText_Solid(font, fitness_text, fitnessColor);
        if (textSurface) {
            SDL_Texture *textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
            if (textTexture) {
                SDL_Rect textRect = {SCREEN_WIDTH - 250, 35 + i * 20, textSurface->w, textSurface->h};
                SDL_RenderCopy(renderer, textTexture, NULL, &textRect);
                SDL_DestroyTexture(textTexture);
            }
            SDL_FreeSurface(textSurface);
        }
    }
}

void draw_minimap(SDL_Renderer *renderer, GameState *game) {
    // Minimap settings
    int minimap_width = 200;
    int minimap_height = 150;
    int minimap_x = SCREEN_WIDTH - minimap_width - 10;
    int minimap_y = SCREEN_HEIGHT - minimap_height - 10;
    
    // Draw minimap background
    SDL_SetRenderDrawColor(renderer, 40, 40, 50, 200);
    SDL_Rect minimap_bg = {minimap_x, minimap_y, minimap_width, minimap_height};
    SDL_RenderFillRect(renderer, &minimap_bg);
    
    // Draw minimap border
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderDrawRect(renderer, &minimap_bg);
    
    // Calculate scale factors
    float scale_x = (float)minimap_width / WORLD_WIDTH;
    float scale_y = (float)minimap_height / WORLD_HEIGHT;
    
    // Draw agents on minimap
    for (int i = 0; i < MAX_AGENTS; i++) {
        if (game->agents[i].alive) {
            int agent_x = minimap_x + (int)(game->agents[i].x * scale_x);
            int agent_y = minimap_y + (int)(game->agents[i].y * scale_y);
            
            // Color based on performance
            if (game->fitness_scores[i] > 100) {
                SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255); // Green for top performers
            } else if (game->fitness_scores[i] > 50) {
                SDL_SetRenderDrawColor(renderer, 255, 255, 0, 255); // Yellow for average
            } else {
                SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // Red for low performers
            }
            
            SDL_Rect agent_dot = {agent_x - 1, agent_y - 1, 2, 2};
            SDL_RenderFillRect(renderer, &agent_dot);
        }
    }
    
    // Draw shapes on minimap
    SDL_SetRenderDrawColor(renderer, 128, 128, 128, 255);
    for (int i = 0; i < 50; i++) {
        if (game->shapes[i].active) {
            int shape_x = minimap_x + (int)(game->shapes[i].x * scale_x);
            int shape_y = minimap_y + (int)(game->shapes[i].y * scale_y);
            SDL_Rect shape_dot = {shape_x, shape_y, 1, 1};
            SDL_RenderFillRect(renderer, &shape_dot);
        }
    }
    
    // Draw camera view rectangle
    float view_width = SCREEN_WIDTH / game->camera.zoom;
    float view_height = SCREEN_HEIGHT / game->camera.zoom;
    int cam_x = minimap_x + (int)((game->camera.x - view_width/2) * scale_x);
    int cam_y = minimap_y + (int)((game->camera.y - view_height/2) * scale_y);
    int cam_w = (int)(view_width * scale_x);
    int cam_h = (int)(view_height * scale_y);
    
    SDL_SetRenderDrawColor(renderer, 255, 255, 0, 255);
    SDL_Rect cam_rect = {cam_x, cam_y, cam_w, cam_h};
    SDL_RenderDrawRect(renderer, &cam_rect);
}
