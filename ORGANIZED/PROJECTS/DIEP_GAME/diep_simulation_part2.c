// Diep.io Tank Simulation - Part 2: Game Mechanics and Rendering
#include "diep_simulation.c"

void handle_input(GameState *game) {
    SDL_Event event;
    
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_QUIT:
                game->game_running = false;
                break;
            case SDL_KEYDOWN:
                switch (event.key.keysym.sym) {
                    case SDLK_ESCAPE:
                    case SDLK_q:
                        game->game_running = false;
                        break;
                    case SDLK_p:
                        game->paused = !game->paused;
                        break;
                    case SDLK_KP_PLUS:
                    case SDLK_EQUALS:
                        game->camera.zoom = fminf(2.0f, game->camera.zoom + 0.1f);
                        break;
                    case SDLK_KP_MINUS:
                    case SDLK_MINUS:
                        game->camera.zoom = fmaxf(0.2f, game->camera.zoom - 0.1f);
                        break;
                    case SDLK_SPACE: {
                        // Focus camera on random agent
                        int focus_agent = rand() % MAX_AGENTS;
                        if (game->agents[focus_agent].alive) {
                            game->camera.target_x = game->agents[focus_agent].x;
                            game->camera.target_y = game->agents[focus_agent].y;
                        }
                        break;
                    }
                }
                break;
        }
    }
}

void update_game(GameState *game) {
    if (game->paused) {
        return;
    }
    
    // Update all agents
    for (int i = 0; i < MAX_AGENTS; i++) {
        if (game->agents[i].alive) {
            update_agent_ai(game, i);
            
            // Update position
            game->agents[i].x += game->agents[i].vx;
            game->agents[i].y += game->agents[i].vy;
            
            // Keep in world bounds
            game->agents[i].x = fmaxf(ARENA_MARGIN + game->agents[i].radius, 
                                     fminf(WORLD_WIDTH - ARENA_MARGIN - game->agents[i].radius, game->agents[i].x));
            game->agents[i].y = fmaxf(ARENA_MARGIN + game->agents[i].radius, 
                                     fminf(WORLD_HEIGHT - ARENA_MARGIN - game->agents[i].radius, game->agents[i].y));
        }
    }
    
    // Update bullets
    update_bullets(game);
    
    // Update shapes
    update_shapes(game);
    
    // Check collisions
    check_collisions(game);
    
    // Update camera
    update_camera(game);
    
    // Update fitness scores
    update_fitness_scores(game);
    
    // Respawn dead agents
    for (int i = 0; i < MAX_AGENTS; i++) {
        if (!game->agents[i].alive) {
            respawn_agent(&game->agents[i], game);
        }
    }
}

void update_camera(GameState *game) {
    // Smooth camera following
    float follow_speed = 0.05f;
    game->camera.x += (game->camera.target_x - game->camera.x) * follow_speed;
    game->camera.y += (game->camera.target_y - game->camera.y) * follow_speed;
    
    // Auto-follow highest scoring agent
    if (game->frame_count % 120 == 0) { // Every 2 seconds
        int best_agent = 0;
        int best_score = -1;
        
        for (int i = 0; i < MAX_AGENTS; i++) {
            if (game->agents[i].alive && game->agents[i].score > best_score) {
                best_score = game->agents[i].score;
                best_agent = i;
            }
        }
        
        if (best_score >= 0) {
            game->camera.target_x = game->agents[best_agent].x;
            game->camera.target_y = game->agents[best_agent].y;
        }
    }
}

float world_to_screen_x(GameState *game, float x) {
    return (x - game->camera.x) * game->camera.zoom + SCREEN_WIDTH / 2;
}

float world_to_screen_y(GameState *game, float y) {
    return (y - game->camera.y) * game->camera.zoom + SCREEN_HEIGHT / 2;
}

bool is_on_screen(GameState *game, float x, float y, float radius) {
    float screen_x = world_to_screen_x(game, x);
    float screen_y = world_to_screen_y(game, y);
    float screen_radius = radius * game->camera.zoom;
    
    return (screen_x + screen_radius >= 0 && screen_x - screen_radius <= SCREEN_WIDTH &&
            screen_y + screen_radius >= 0 && screen_y - screen_radius <= SCREEN_HEIGHT);
}

float get_distance(float x1, float y1, float x2, float y2) {
    return sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

uint32_t get_current_time_ms() {
    return (uint32_t)SDL_GetTicks();
}

void update_agent_stats(Agent *agent) {
    TankStats base_stats = TANK_STATS[agent->tank_type];
    
    // Apply level bonuses
    float level_multiplier = 1.0f + (agent->level - 1) * 0.1f;
    
    agent->stats.speed = base_stats.speed * level_multiplier;
    agent->stats.bullet_speed = base_stats.bullet_speed * level_multiplier;
    agent->stats.bullet_damage = base_stats.bullet_damage * level_multiplier;
    agent->stats.fire_rate = base_stats.fire_rate;
    agent->stats.bullet_count = base_stats.bullet_count;
    agent->stats.spread_angle = base_stats.spread_angle;
    
    agent->max_health = 100 + (agent->level - 1) * 20;
}

void add_experience(Agent *agent, int amount) {
    agent->experience += amount;
    
    while (agent->experience >= agent->experience_to_next_level) {
        agent->experience -= agent->experience_to_next_level;
        level_up(agent);
    }
}

void level_up(Agent *agent) {
    agent->level++;
    agent->experience_to_next_level = agent->level * 100;
    agent->health = agent->max_health;
    
    // Evolve tank at certain levels
    TankType new_type = agent->tank_type;
    switch (agent->level) {
        case 5:
            new_type = TANK_TWIN;
            break;
        case 10:
            new_type = TANK_SNIPER;
            break;
        case 15:
            new_type = TANK_MACHINE;
            break;
        case 20:
            new_type = TANK_DESTROYER;
            break;
    }
    
    if (new_type != agent->tank_type) {
        evolve_tank(agent, new_type);
    }
    
    update_agent_stats(agent);
}

void evolve_tank(Agent *agent, TankType new_type) {
    agent->tank_type = new_type;
    
    // Adjust appearance based on tank type
    switch (new_type) {
        case TANK_TWIN:
            agent->radius = TANK_SIZE * 1.1f;
            break;
        case TANK_SNIPER:
            agent->radius = TANK_SIZE * 0.9f;
            break;
        case TANK_MACHINE:
            agent->radius = TANK_SIZE * 1.2f;
            break;
        case TANK_DESTROYER:
            agent->radius = TANK_SIZE * 1.4f;
            break;
        default:
            agent->radius = TANK_SIZE;
            break;
    }
}

void fire_bullet(GameState *game, int agent_index) {
    uint32_t current_time = get_current_time_ms();
    Agent *agent = &game->agents[agent_index];
    
    if (!agent->alive || current_time - agent->last_fire_time < agent->stats.fire_rate) {
        return;
    }
    
    // Fire bullets based on tank type
    for (int i = 0; i < agent->stats.bullet_count; i++) {
        // Find inactive bullet
        for (int j = 0; j < MAX_BULLETS; j++) {
            if (!game->bullets[j].active) {
                float angle_offset = 0;
                if (agent->stats.bullet_count > 1) {
                    angle_offset = (i - (agent->stats.bullet_count - 1) / 2.0f) * agent->stats.spread_angle;
                }
                
                float fire_angle = agent->angle + angle_offset;
                game->bullets[j].x = agent->x + cosf(fire_angle) * (agent->radius + 15);
                game->bullets[j].y = agent->y + sinf(fire_angle) * (agent->radius + 15);
                game->bullets[j].vx = cosf(fire_angle) * agent->stats.bullet_speed;
                game->bullets[j].vy = sinf(fire_angle) * agent->stats.bullet_speed;
                game->bullets[j].radius = BULLET_SIZE;
                game->bullets[j].color = agent->color;
                game->bullets[j].active = true;
                game->bullets[j].owner_id = agent_index;
                game->bullets[j].damage = (int)agent->stats.bullet_damage;
                game->bullets[j].entity_type = ENTITY_BULLET_SELF;
                
                break;
            }
        }
    }
    
    agent->last_fire_time = current_time;
}

void update_bullets(GameState *game) {
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (game->bullets[i].active) {
            game->bullets[i].x += game->bullets[i].vx;
            game->bullets[i].y += game->bullets[i].vy;
            
            // Check if bullet is out of world bounds
            if (game->bullets[i].x < ARENA_MARGIN || 
                game->bullets[i].x > WORLD_WIDTH - ARENA_MARGIN ||
                game->bullets[i].y < ARENA_MARGIN || 
                game->bullets[i].y > WORLD_HEIGHT - ARENA_MARGIN) {
                game->bullets[i].active = false;
            }
        }
    }
}

void update_shapes(GameState *game) {
    for (int i = 0; i < 50; i++) {
        if (!game->shapes[i].active) {
            // Respawn shape
            game->shapes[i].x = (float)(rand() % (WORLD_WIDTH - 200)) + 100;
            game->shapes[i].y = (float)(rand() % (WORLD_HEIGHT - 200)) + 100;
            game->shapes[i].radius = (float)(rand() % 20) + 10;
            game->shapes[i].active = true;
            game->shapes[i].health = (int)game->shapes[i].radius;
            game->shapes[i].experience_value = (int)game->shapes[i].radius * 2;
        }
    }
}

void check_collisions(GameState *game) {
    // Check bullet-agent collisions
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (!game->bullets[i].active) continue;
        
        Bullet *bullet = &game->bullets[i];
        
        // Check collision with agents
        for (int j = 0; j < MAX_AGENTS; j++) {
            if (bullet->owner_id != j && game->agents[j].alive) {
                float dist = get_distance(bullet->x, bullet->y, 
                                       game->agents[j].x, game->agents[j].y);
                if (dist < bullet->radius + game->agents[j].radius) {
                    game->agents[j].health -= bullet->damage;
                    bullet->active = false;
                    
                    if (game->agents[j].health <= 0) {
                        game->agents[j].alive = false;
                        game->agents[bullet->owner_id].score += 10;
                        add_experience(&game->agents[bullet->owner_id], 30);
                    }
                }
            }
        }
    }
    
    // Check bullet-shape collisions
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (!game->bullets[i].active) continue;
        
        Bullet *bullet = &game->bullets[i];
        
        for (int j = 0; j < 50; j++) {
            if (!game->shapes[j].active) continue;
            
            float dist = get_distance(bullet->x, bullet->y, 
                                   game->shapes[j].x, game->shapes[j].y);
            if (dist < bullet->radius + game->shapes[j].radius) {
                game->shapes[j].health -= bullet->damage;
                bullet->active = false;
                
                if (game->shapes[j].health <= 0) {
                    game->shapes[j].active = false;
                    add_experience(&game->agents[bullet->owner_id], game->shapes[j].experience_value);
                }
            }
        }
    }
}

void respawn_agent(Agent *agent, GameState *game) {
    (void)game; // Suppress unused parameter warning
    static uint32_t respawn_time[MAX_AGENTS] = {0};
    uint32_t current_time = get_current_time_ms();
    
    if (respawn_time[agent->agent_id] == 0) {
        respawn_time[agent->agent_id] = current_time;
        return;
    }
    
    if (current_time - respawn_time[agent->agent_id] > 2000) { // 2 second respawn
        agent->x = (float)(rand() % (WORLD_WIDTH - 400)) + 200;
        agent->y = (float)(rand() % (WORLD_HEIGHT - 400)) + 200;
        agent->vx = 0;
        agent->vy = 0;
        agent->angle = (float)(rand() % 360) * M_PI / 180.0f;
        agent->health = agent->max_health;
        agent->alive = true;
        respawn_time[agent->agent_id] = 0;
    }
}

void update_fitness_scores(GameState *game) {
    for (int i = 0; i < MAX_AGENTS; i++) {
        Agent *agent = &game->agents[i];
        
        // Fitness based on score, level, and survival time
        float score_fitness = agent->score * 1.0f;
        float level_fitness = agent->level * 50.0f;
        float survival_fitness = (agent->alive ? 1.0f : 0.0f) * 100.0f;
        
        game->fitness_scores[i] = score_fitness + level_fitness + survival_fitness;
    }
}

void render_game(GameState *game) {
    // Clear screen
    SDL_SetRenderDrawColor(game->renderer, 20, 20, 30, 255);
    SDL_RenderClear(game->renderer);
    
    // Draw grid lines (for debugging AI observation)
    if (game->paused) {
        SDL_SetRenderDrawColor(game->renderer, 40, 40, 50, 255);
        
        for (int x = 0; x <= GRID_WIDTH; x++) {
            float world_x = ARENA_MARGIN + x * (WORLD_WIDTH - 2 * ARENA_MARGIN) / GRID_WIDTH;
            float screen_x1 = world_to_screen_x(game, world_x);
            float screen_y1 = world_to_screen_y(game, ARENA_MARGIN);
            float screen_y2 = world_to_screen_y(game, WORLD_HEIGHT - ARENA_MARGIN);
            SDL_RenderDrawLine(game->renderer, (int)screen_x1, (int)screen_y1, 
                             (int)screen_x1, (int)screen_y2);
        }
        for (int y = 0; y <= GRID_HEIGHT; y++) {
            float world_y = ARENA_MARGIN + y * (WORLD_HEIGHT - 2 * ARENA_MARGIN) / GRID_HEIGHT;
            float screen_x1 = world_to_screen_x(game, ARENA_MARGIN);
            float screen_x2 = world_to_screen_x(game, WORLD_WIDTH - ARENA_MARGIN);
            float screen_y1 = world_to_screen_y(game, world_y);
            SDL_RenderDrawLine(game->renderer, (int)screen_x1, (int)screen_y1, 
                             (int)screen_x2, (int)screen_y1);
        }
    }
    
    // Draw world boundaries
    SDL_SetRenderDrawColor(game->renderer, 128, 128, 128, 255);
    SDL_Rect world_bounds = {
        (int)world_to_screen_x(game, ARENA_MARGIN),
        (int)world_to_screen_y(game, ARENA_MARGIN),
        (int)((WORLD_WIDTH - 2 * ARENA_MARGIN) * game->camera.zoom),
        (int)((WORLD_HEIGHT - 2 * ARENA_MARGIN) * game->camera.zoom)
    };
    SDL_RenderDrawRect(game->renderer, &world_bounds);
    
    // Draw shapes
    for (int i = 0; i < 50; i++) {
        if (game->shapes[i].active && is_on_screen(game, game->shapes[i].x, game->shapes[i].y, game->shapes[i].radius)) {
            draw_shape(game, &game->shapes[i]);
        }
    }
    
    // Draw agents
    for (int i = 0; i < MAX_AGENTS; i++) {
        if (game->agents[i].alive && is_on_screen(game, game->agents[i].x, game->agents[i].y, game->agents[i].radius)) {
            draw_agent(game, &game->agents[i]);
        }
    }
    
    // Draw bullets
    for (int i = 0; i < MAX_BULLETS; i++) {
        if (game->bullets[i].active && is_on_screen(game, game->bullets[i].x, game->bullets[i].y, game->bullets[i].radius)) {
            draw_bullet(game, &game->bullets[i]);
        }
    }
    
    // Draw UI
    draw_ui(game->renderer, game->font, game);
    
    // Draw minimap
    draw_minimap(game->renderer, game);
    
    // Present
    SDL_RenderPresent(game->renderer);
}

void cleanup_game(GameState *game) {
    if (game->font) {
        TTF_CloseFont(game->font);
        game->font = NULL;
    }
    if (game->renderer) {
        SDL_DestroyRenderer(game->renderer);
        game->renderer = NULL;
    }
    if (game->window) {
        SDL_DestroyWindow(game->window);
        game->window = NULL;
    }
    
    TTF_Quit();
    SDL_Quit();
}
