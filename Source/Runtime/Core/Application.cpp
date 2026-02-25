#include "Application.hpp"
#include "../Renderer/RenderSystem.hpp"

namespace engine {

    Application::Application() {
        
        //basic systems
        
        //AddSystem<WindowSystem>();
        //AddSystem<InputSystem>();
        //AddSystem<SoundSystem>();

        //world systems
 
        //AddSystem<TerrainSystem>();
        //AddSystem<SceneManager>();

        //get input and calculate data
        
        //AddSystem<PhysicsSystem>();

        //final render 
        
        //AddSystem<CameraSystem>();
        //AddSystem<UISystem>();
        AddSystem<RenderSystem>(Running);


        // left or right Init
        for (auto& sys : Systems) 
            sys->Init();
        
    }

    Application::~Application() {
		// right to left Shutdown
        for (auto it = Systems.rbegin(); it != Systems.rend(); ++it) 
            (*it)->Shutdown();
        // auto clear unique_ptrs in vector
        Systems.clear();
    }

    void Application::Run() {
        while (Running) {
            float dt = CalcDeltaTime();
            for (auto& sys : Systems)
                sys->Update(dt);
        }
    }

}